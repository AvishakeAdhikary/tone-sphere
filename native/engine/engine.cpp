#include "engine.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <deque>
#include <new>
#include <numbers>

#include "rt_alloc.h"

namespace ts {

namespace {

using Clock = std::chrono::steady_clock;

const ts_port_buffer* find_port(const ts_port_buffer* ports, uint32_t count, uint32_t node_id) noexcept {
    for (uint32_t i = 0; i < count; ++i)
        if (ports[i].node_id == node_id) return &ports[i];
    return nullptr;
}

void relaxed_max(std::atomic<uint64_t>& a, uint64_t v) noexcept {
    if (v > a.load(std::memory_order_relaxed)) a.store(v, std::memory_order_relaxed);
}

void relaxed_min(std::atomic<uint64_t>& a, uint64_t v) noexcept {
    if (v < a.load(std::memory_order_relaxed)) a.store(v, std::memory_order_relaxed);
}

void relaxed_add(std::atomic<uint64_t>& a, uint64_t v) noexcept {
    a.store(a.load(std::memory_order_relaxed) + v, std::memory_order_relaxed);
}

}  // namespace

Engine::Engine(uint32_t sample_rate, uint32_t max_block)
    : sample_rate_(sample_rate),
      max_block_(max_block),
      // 64 buckets spanning two full periods of the largest block: anything past the end
      // is already a dropout-grade overrun and lands in the last bucket.
      bucket_ns_(std::max<uint64_t>(1, (2ull * max_block * 1'000'000'000ull / sample_rate) / TS_HISTOGRAM_BUCKETS)) {}

Engine::~Engine() {
    Plan* last = current_.exchange(nullptr);
    if (last) retired_.push_back(last);
    for (Plan* p : retired_) delete p;
}

ts_result Engine::apply_plan(const ts_node_desc* nodes, uint32_t node_count,
                             const ts_route_desc* routes, uint32_t route_count) {
    if (node_count > TS_MAX_NODES) { fail("too many nodes"); return TS_ERR_INVALID; }
    if (route_count > TS_MAX_ROUTES) { fail("too many routes"); return TS_ERR_INVALID; }
    if ((node_count && !nodes) || (route_count && !routes)) { fail("null plan array"); return TS_ERR_INVALID; }

    std::unordered_map<uint32_t, uint32_t> by_id;
    by_id.reserve(node_count);
    for (uint32_t i = 0; i < node_count; ++i) {
        const ts_node_desc& n = nodes[i];
        if (n.kind != TS_NODE_SOURCE && n.kind != TS_NODE_BUS && n.kind != TS_NODE_SINK) {
            fail("node " + std::to_string(n.id) + ": unknown kind " + std::to_string(n.kind));
            return TS_ERR_INVALID;
        }
        if (n.channels < 1 || n.channels > TS_MAX_CHANNELS) {
            fail("node " + std::to_string(n.id) + ": channel count " + std::to_string(n.channels) + " out of range");
            return TS_ERR_INVALID;
        }
        if ((n.flags & TS_NODE_FLAG_RING) && n.kind == TS_NODE_BUS) {
            fail("node " + std::to_string(n.id) + ": a bus cannot be a ring port");
            return TS_ERR_INVALID;
        }
        if ((n.flags & TS_NODE_FLAG_RING) && (n.ring_frames < max_block_ || n.ring_frames > (1u << 24))) {
            fail("node " + std::to_string(n.id) + ": ring must hold at least one block and at most 2^24 frames");
            return TS_ERR_INVALID;
        }
        if (!by_id.emplace(n.id, i).second) {
            fail("duplicate node id " + std::to_string(n.id));
            return TS_ERR_INVALID;
        }
    }

    std::vector<std::vector<uint32_t>> incoming(node_count);
    std::vector<std::vector<uint32_t>> outgoing(node_count);
    std::unordered_map<uint64_t, uint32_t> seen_routes;
    for (uint32_t r = 0; r < route_count; ++r) {
        const ts_route_desc& d = routes[r];
        auto s = by_id.find(d.source);
        auto t = by_id.find(d.dest);
        const std::string label = "route " + std::to_string(d.source) + "->" + std::to_string(d.dest);
        if (s == by_id.end() || t == by_id.end()) { fail(label + ": unknown node"); return TS_ERR_INVALID; }
        if (nodes[s->second].kind == TS_NODE_SINK) { fail(label + ": a sink cannot feed anything"); return TS_ERR_INVALID; }
        if (nodes[t->second].kind == TS_NODE_SOURCE) { fail(label + ": a source cannot be fed"); return TS_ERR_INVALID; }
        if (d.source == d.dest) { fail(label + ": routes a node to itself"); return TS_ERR_CYCLE; }
        if (!std::isfinite(d.gain) || !std::isfinite(d.pan)) { fail(label + ": non-finite gain or pan"); return TS_ERR_INVALID; }
        if (!seen_routes.emplace(route_key(d.source, d.dest), r).second) { fail(label + ": duplicate route"); return TS_ERR_INVALID; }
        incoming[t->second].push_back(r);
        outgoing[s->second].push_back(t->second);
    }

    // Kahn's algorithm. Anything left unordered is on a cycle; the audio thread only ever
    // sees a plan in which every node's inputs are computed before the node itself.
    std::vector<uint32_t> order;
    order.reserve(node_count);
    std::vector<uint32_t> indegree(node_count);
    for (uint32_t i = 0; i < node_count; ++i) indegree[i] = static_cast<uint32_t>(incoming[i].size());
    std::deque<uint32_t> ready;
    for (uint32_t i = 0; i < node_count; ++i)
        if (indegree[i] == 0) ready.push_back(i);
    while (!ready.empty()) {
        const uint32_t i = ready.front();
        ready.pop_front();
        order.push_back(i);
        for (uint32_t t : outgoing[i])
            if (--indegree[t] == 0) ready.push_back(t);
    }
    if (order.size() != node_count) {
        std::string on_cycle;
        for (uint32_t i = 0; i < node_count; ++i)
            if (indegree[i]) on_cycle += (on_cycle.empty() ? "" : ", ") + std::to_string(nodes[i].id);
        fail("feedback loop through node(s) " + on_cycle);
        return TS_ERR_CYCLE;
    }

    auto plan = std::make_unique<Plan>();
    plan->node_count = node_count;
    plan->route_count = route_count;
    plan->nodes.reset(new Node[node_count]);
    plan->routes.reset(new Route[route_count]);

    size_t total_channels = 0;
    for (uint32_t i = 0; i < node_count; ++i) total_channels += nodes[i].channels;
    plan->storage.reset(new float[std::max<size_t>(1, total_channels * max_block_)]());
    plan->scratch.reset(new float[static_cast<size_t>(TS_MAX_CHANNELS) * max_block_]());

    std::unordered_map<uint32_t, std::shared_ptr<FrameRing>> rings;
    std::vector<uint32_t> position(node_count);
    for (uint32_t k = 0; k < node_count; ++k) position[order[k]] = k;

    const uint32_t meter_generation = meter_reset_generation_.load(std::memory_order_relaxed);
    float* cursor = plan->storage.get();
    uint32_t route_cursor = 0;
    for (uint32_t k = 0; k < node_count; ++k) {
        const uint32_t i = order[k];
        const ts_node_desc& d = nodes[i];
        Node& n = plan->nodes[k];
        n.id = d.id;
        n.kind = d.kind;
        n.channels = d.channels;
        n.flags = d.flags;
        n.buffer = cursor;
        cursor += static_cast<size_t>(d.channels) * max_block_;
        n.meter.reset_seen = meter_generation;

        if (d.flags & TS_NODE_FLAG_RING) {
            // A ring survives a plan change when nothing about it changed, so audio already
            // queued by a producer is not thrown away because an unrelated route moved.
            std::shared_ptr<FrameRing> ring;
            auto existing = rings_.find(d.id);
            if (existing != rings_.end() && existing->second->channels() == d.channels &&
                existing->second->capacity() == next_pow2(d.ring_frames))
                ring = existing->second;
            else
                ring = std::make_shared<FrameRing>(d.ring_frames, d.channels);
            n.ring = ring.get();
            plan->rings.push_back(ring);
            rings.emplace(d.id, std::move(ring));
        }

        n.route_begin = route_cursor;
        for (uint32_t r : incoming[i]) {
            const ts_route_desc& rd = routes[r];
            Route& route = plan->routes[route_cursor];
            route.source_index = position[by_id[rd.source]];
            route.key = route_key(rd.source, rd.dest);
            route.invert = (rd.flags & TS_ROUTE_FLAG_INVERT) != 0;

            const float pan = std::clamp(rd.pan, -1.0f, 1.0f);
            const float theta = (pan + 1.0f) * static_cast<float>(std::numbers::pi) / 4.0f;
            route.pan_left = std::cos(theta);
            route.pan_right = std::sin(theta);
            // Balance keeps unity at centre and only ever attenuates the far side, so a
            // stereo source does not jump 3 dB the moment it leaves centre.
            const float far = std::cos(std::abs(pan) * static_cast<float>(std::numbers::pi) / 2.0f);
            route.balance_left = pan > 0.0f ? far : 1.0f;
            route.balance_right = pan < 0.0f ? far : 1.0f;

            route.target.store(rd.gain, std::memory_order_relaxed);
            route.muted.store((rd.flags & TS_ROUTE_FLAG_MUTED) ? 1u : 0u, std::memory_order_relaxed);
            float start = 0.0f;
            if (latest_) {
                auto previous = latest_->route_index.find(route.key);
                if (previous != latest_->route_index.end())
                    start = latest_->routes[previous->second].published.load(std::memory_order_relaxed);
            }
            route.current = start;
            route.published.store(start, std::memory_order_relaxed);
            plan->route_index.emplace(route.key, route_cursor);
            ++route_cursor;
        }
        n.route_end = route_cursor;
        plan->node_index.emplace(d.id, k);
    }

    plan->generation = next_generation_++;
    Plan* published = plan.release();
    Plan* old = current_.exchange(published, std::memory_order_seq_cst);
    if (old) retired_.push_back(old);
    latest_ = published;
    rings_ = std::move(rings);
    collect();
    return TS_OK;
}

void Engine::collect() {
    Plan* in_use = hazard_.load(std::memory_order_seq_cst);
    auto keep = std::remove_if(retired_.begin(), retired_.end(), [&](Plan* p) {
        if (p == in_use) return false;
        delete p;
        return true;
    });
    retired_.erase(keep, retired_.end());
}

Plan* Engine::acquire() noexcept {
    Plan* p = current_.load(std::memory_order_acquire);
    for (;;) {
        hazard_.store(p, std::memory_order_seq_cst);
        Plan* again = current_.load(std::memory_order_seq_cst);
        if (again == p) return p;
        p = again;
    }
}

void Engine::release() noexcept { hazard_.store(nullptr, std::memory_order_release); }

ts_result Engine::set_route_gain(uint32_t source, uint32_t dest, float gain) {
    if (!std::isfinite(gain)) { fail("non-finite gain"); return TS_ERR_INVALID; }
    if (!latest_) { fail("no plan applied"); return TS_ERR_STATE; }
    auto it = latest_->route_index.find(route_key(source, dest));
    if (it == latest_->route_index.end()) { fail("no such route"); return TS_ERR_NOT_FOUND; }
    latest_->routes[it->second].target.store(gain, std::memory_order_relaxed);
    return TS_OK;
}

ts_result Engine::set_route_muted(uint32_t source, uint32_t dest, bool muted) {
    if (!latest_) { fail("no plan applied"); return TS_ERR_STATE; }
    auto it = latest_->route_index.find(route_key(source, dest));
    if (it == latest_->route_index.end()) { fail("no such route"); return TS_ERR_NOT_FOUND; }
    latest_->routes[it->second].muted.store(muted ? 1u : 0u, std::memory_order_relaxed);
    return TS_OK;
}

ts_result Engine::set_master_gain(float gain) {
    if (!std::isfinite(gain) || gain < 0.0f) { fail("master gain must be finite and non-negative"); return TS_ERR_INVALID; }
    master_target_.store(gain, std::memory_order_relaxed);
    return TS_OK;
}

ts_result Engine::process_offline(const ts_port_buffer* inputs, uint32_t input_count,
                                  ts_port_buffer* outputs, uint32_t output_count, uint32_t frames) {
    if (backend_running_.load(std::memory_order_acquire)) {
        fail("a device backend is running; the engine already has an audio thread");
        return TS_ERR_STATE;
    }
    if (frames == 0 || frames > max_block_) {
        fail("frames must be 1.." + std::to_string(max_block_));
        return TS_ERR_INVALID;
    }
    for (uint32_t i = 0; i < input_count; ++i)
        if (!inputs[i].data || inputs[i].channels == 0) { fail("input port without data"); return TS_ERR_INVALID; }
    for (uint32_t i = 0; i < output_count; ++i)
        if (!outputs[i].data || outputs[i].channels == 0) { fail("output port without data"); return TS_ERR_INVALID; }
    run_block(inputs, input_count, outputs, output_count, frames);
    return TS_OK;
}

void Engine::run_block(const ts_port_buffer* inputs, uint32_t input_count,
                       ts_port_buffer* outputs, uint32_t output_count, uint32_t frames) noexcept {
    AudioThreadScope scope;
    const auto started = Clock::now();

    const uint32_t stats_generation = stats_reset_generation_.load(std::memory_order_acquire);
    if (stats_generation != stats_reset_seen_) {
        stats_reset_seen_ = stats_generation;
        blocks_.store(0, std::memory_order_relaxed);
        xruns_.store(0, std::memory_order_relaxed);
        overruns_.store(0, std::memory_order_relaxed);
        underruns_.store(0, std::memory_order_relaxed);
        ns_min_.store(UINT64_MAX, std::memory_order_relaxed);
        ns_max_.store(0, std::memory_order_relaxed);
        ns_total_.store(0, std::memory_order_relaxed);
        for (auto& bucket : histogram_) bucket.store(0, std::memory_order_relaxed);
        rt_allocations_baseline_ = g_rt_allocations.load(std::memory_order_relaxed);
    }

    Plan* plan = acquire();
    if (plan) {
        process_plan(*plan, inputs, input_count, outputs, output_count, frames);
        plan_generation_.store(plan->generation, std::memory_order_relaxed);
    } else {
        for (uint32_t i = 0; i < output_count; ++i)
            std::memset(outputs[i].data, 0, sizeof(float) * frames * outputs[i].channels);
    }
    release();

    const uint64_t elapsed = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - started).count());
    relaxed_add(blocks_, 1);
    relaxed_add(ns_total_, elapsed);
    relaxed_min(ns_min_, elapsed);
    relaxed_max(ns_max_, elapsed);
    const uint64_t bucket = std::min<uint64_t>(elapsed / bucket_ns_, TS_HISTOGRAM_BUCKETS - 1);
    relaxed_add(histogram_[bucket], 1);
    period_ns_.store(static_cast<uint64_t>(frames) * 1'000'000'000ull / sample_rate_, std::memory_order_relaxed);
    rt_allocations_.store(g_rt_allocations.load(std::memory_order_relaxed) - rt_allocations_baseline_,
                          std::memory_order_relaxed);
}

void Engine::process_plan(Plan& plan, const ts_port_buffer* inputs, uint32_t input_count,
                          ts_port_buffer* outputs, uint32_t output_count, uint32_t frames) noexcept {
    const uint32_t meter_generation = meter_reset_generation_.load(std::memory_order_acquire);
    const float master_from = master_current_;
    const float master_to = master_target_.load(std::memory_order_relaxed);

    for (uint32_t k = 0; k < plan.node_count; ++k) {
        Node& node = plan.nodes[k];
        if (node.kind == TS_NODE_SOURCE)
            fill_source(plan, node, inputs, input_count, frames);
        else
            mix_into(plan, node, frames);

        if (node.kind == TS_NODE_SINK)
            emit_sink(plan, node, outputs, output_count, frames, master_from, master_to);
        meter(node, frames, meter_generation);
    }
    master_current_ = master_to;

    // Offline outputs that no sink claimed must still come back as silence, not as
    // whatever the caller's buffer happened to hold.
    for (uint32_t i = 0; i < output_count; ++i) {
        bool claimed = false;
        for (uint32_t k = 0; k < plan.node_count; ++k) {
            const Node& node = plan.nodes[k];
            if (node.kind == TS_NODE_SINK && !(node.flags & TS_NODE_FLAG_RING) && node.id == outputs[i].node_id) {
                claimed = true;
                break;
            }
        }
        if (!claimed) std::memset(outputs[i].data, 0, sizeof(float) * frames * outputs[i].channels);
    }
}

void Engine::fill_source(Plan& plan, Node& node, const ts_port_buffer* inputs, uint32_t input_count,
                         uint32_t frames) noexcept {
    const float* interleaved = nullptr;
    uint32_t stride = 0;

    if (node.ring) {
        float* staging = plan.scratch.get();
        const uint32_t got = node.ring->read(staging, frames);
        if (got < frames) {
            std::memset(staging + static_cast<size_t>(got) * node.channels, 0,
                        sizeof(float) * (frames - got) * node.channels);
            relaxed_add(underruns_, frames - got);
            if (!node.ring_trouble) post(TS_EVENT_RING_UNDERRUN, node.id, frames - got);
            node.ring_trouble = true;
        } else {
            node.ring_trouble = false;
        }
        interleaved = staging;
        stride = node.channels;
    } else if (const ts_port_buffer* port = find_port(inputs, input_count, node.id)) {
        interleaved = port->data;
        stride = port->channels;
    }

    for (uint32_t c = 0; c < node.channels; ++c) {
        float* dst = node.buffer + static_cast<size_t>(c) * max_block_;
        if (!interleaved || c >= stride) {
            std::memset(dst, 0, sizeof(float) * frames);
            continue;
        }
        for (uint32_t i = 0; i < frames; ++i) dst[i] = interleaved[static_cast<size_t>(i) * stride + c];
    }
}

void Engine::mix_into(Plan& plan, Node& node, uint32_t frames) noexcept {
    for (uint32_t c = 0; c < node.channels; ++c)
        std::memset(node.buffer + static_cast<size_t>(c) * max_block_, 0, sizeof(float) * frames);

    const float inv_frames = 1.0f / static_cast<float>(frames);
    for (uint32_t r = node.route_begin; r < node.route_end; ++r) {
        Route& route = plan.routes[r];
        const Node& src = plan.nodes[route.source_index];
        const float from = route.current;
        const float to = route.muted.load(std::memory_order_relaxed) ? 0.0f : route.target.load(std::memory_order_relaxed);
        route.current = to;
        route.published.store(to, std::memory_order_relaxed);
        if (from == 0.0f && to == 0.0f) continue;

        const float sign = route.invert ? -1.0f : 1.0f;
        const uint32_t S = src.channels;
        const uint32_t D = node.channels;

        for (uint32_t c = 0; c < D; ++c) {
            float* dst = node.buffer + static_cast<size_t>(c) * max_block_;
            // Channel mapping, one rule per shape, matching the legacy host's behaviour
            // except where that was a bug (stereo balance, see Route).
            float channel_gain = 1.0f;
            int32_t single = -1;
            bool average = false;
            if (S == 1 && D == 2) {
                single = 0;
                channel_gain = c == 0 ? route.pan_left : route.pan_right;
            } else if (S == D) {
                single = static_cast<int32_t>(c);
                if (D == 2) channel_gain = c == 0 ? route.balance_left : route.balance_right;
            } else if (S == 1) {
                single = 0;
            } else if (D == 1) {
                average = true;
                channel_gain = 1.0f / static_cast<float>(S);
            } else if (c < S) {
                single = static_cast<int32_t>(c);
            } else {
                continue;
            }

            const float g0 = from * channel_gain * sign;
            const float g1 = to * channel_gain * sign;
            const float step = (g1 - g0) * inv_frames;
            if (average) {
                for (uint32_t s = 0; s < S; ++s) {
                    const float* in = src.buffer + static_cast<size_t>(s) * max_block_;
                    if (g0 == g1)
                        for (uint32_t i = 0; i < frames; ++i) dst[i] += in[i] * g1;
                    else
                        for (uint32_t i = 0; i < frames; ++i) dst[i] += in[i] * (g0 + step * static_cast<float>(i + 1));
                }
            } else {
                const float* in = src.buffer + static_cast<size_t>(single) * max_block_;
                if (g0 == g1)
                    for (uint32_t i = 0; i < frames; ++i) dst[i] += in[i] * g1;
                else
                    for (uint32_t i = 0; i < frames; ++i) dst[i] += in[i] * (g0 + step * static_cast<float>(i + 1));
            }
        }
    }
}

void Engine::emit_sink(Plan& plan, Node& node, ts_port_buffer* outputs, uint32_t output_count,
                       uint32_t frames, float master_from, float master_to) noexcept {
    const float step = (master_to - master_from) / static_cast<float>(frames);
    bool bad = false;
    for (uint32_t c = 0; c < node.channels; ++c) {
        float* buf = node.buffer + static_cast<size_t>(c) * max_block_;
        for (uint32_t i = 0; i < frames; ++i) {
            buf[i] *= master_from + step * static_cast<float>(i + 1);
            if (!std::isfinite(buf[i])) bad = true;
        }
    }
    // A NaN reaching a driver is a burst of full-scale noise in someone's headphones.
    // Silence the block, say so once, and keep running.
    if (bad) {
        for (uint32_t c = 0; c < node.channels; ++c)
            std::memset(node.buffer + static_cast<size_t>(c) * max_block_, 0, sizeof(float) * frames);
        if (!node.nonfinite) post(TS_EVENT_NONFINITE, node.id, 0);
    }
    node.nonfinite = bad;

    float* interleaved = nullptr;
    uint32_t stride = 0;
    if (node.ring) {
        interleaved = plan.scratch.get();
        stride = node.channels;
    } else {
        for (uint32_t i = 0; i < output_count; ++i)
            if (outputs[i].node_id == node.id) {
                interleaved = outputs[i].data;
                stride = outputs[i].channels;
                break;
            }
    }
    if (!interleaved) return;

    for (uint32_t i = 0; i < frames; ++i)
        for (uint32_t c = 0; c < stride; ++c)
            interleaved[static_cast<size_t>(i) * stride + c] =
                c < node.channels ? node.buffer[static_cast<size_t>(c) * max_block_ + i] : 0.0f;

    if (node.ring) {
        const uint32_t written = node.ring->write(interleaved, frames);
        if (written < frames) {
            relaxed_add(overruns_, frames - written);
            if (!node.ring_trouble) post(TS_EVENT_RING_OVERRUN, node.id, frames - written);
            node.ring_trouble = true;
        } else {
            node.ring_trouble = false;
        }
    }
}

void Engine::meter(Node& node, uint32_t frames, uint32_t reset_generation) noexcept {
    MeterState& m = node.meter;
    if (m.reset_seen != reset_generation) {
        m.reset_seen = reset_generation;
        m.peak_since_reset = 0.0f;
        m.clipped.store(0, std::memory_order_relaxed);
    }
    float block_peak = 0.0f;
    double sum_squares = 0.0;
    for (uint32_t c = 0; c < node.channels; ++c) {
        const float* buf = node.buffer + static_cast<size_t>(c) * max_block_;
        for (uint32_t i = 0; i < frames; ++i) {
            const float a = std::abs(buf[i]);
            block_peak = a > block_peak ? a : block_peak;
            sum_squares += static_cast<double>(buf[i]) * buf[i];
        }
    }
    if (block_peak > m.peak_since_reset) m.peak_since_reset = block_peak;
    m.peak.store(m.peak_since_reset, std::memory_order_relaxed);
    m.rms.store(static_cast<float>(std::sqrt(sum_squares / (static_cast<double>(frames) * node.channels))),
                std::memory_order_relaxed);
    if (block_peak >= 1.0f) m.clipped.store(1, std::memory_order_relaxed);
}

void Engine::post(uint32_t code, uint32_t arg0, uint64_t arg1) noexcept {
    const uint64_t block = blocks_.load(std::memory_order_relaxed);
    if (events_lost_) {
        if (!events_.push(ts_event{TS_EVENT_EVENTS_LOST, events_lost_, 0, block})) {
            ++events_lost_;
            return;
        }
        events_lost_ = 0;
    }
    if (!events_.push(ts_event{code, arg0, arg1, block})) ++events_lost_;
}

FrameRing* Engine::find_ring(uint32_t node_id, uint32_t kind) {
    if (!latest_) { fail("no plan applied"); return nullptr; }
    auto it = latest_->node_index.find(node_id);
    if (it == latest_->node_index.end()) { fail("no such node"); return nullptr; }
    const Node& node = latest_->nodes[it->second];
    if (!node.ring || node.kind != kind) {
        fail(kind == TS_NODE_SOURCE ? "node is not a ring source" : "node is not a ring sink");
        return nullptr;
    }
    return node.ring;
}

int32_t Engine::port_write(uint32_t node_id, const float* data, uint32_t frames) {
    FrameRing* ring = find_ring(node_id, TS_NODE_SOURCE);
    if (!ring) return TS_ERR_NOT_FOUND;
    return static_cast<int32_t>(ring->write(data, frames));
}

int32_t Engine::port_read(uint32_t node_id, float* data, uint32_t frames) {
    FrameRing* ring = find_ring(node_id, TS_NODE_SINK);
    if (!ring) return TS_ERR_NOT_FOUND;
    return static_cast<int32_t>(ring->read(data, frames));
}

int32_t Engine::port_available(uint32_t node_id) {
    if (!latest_) { fail("no plan applied"); return TS_ERR_STATE; }
    auto it = latest_->node_index.find(node_id);
    if (it == latest_->node_index.end() || !latest_->nodes[it->second].ring) { fail("not a ring node"); return TS_ERR_NOT_FOUND; }
    return static_cast<int32_t>(latest_->nodes[it->second].ring->available());
}

void Engine::get_stats(ts_stats& out) const {
    std::memset(&out, 0, sizeof(out));
    out.blocks = blocks_.load(std::memory_order_relaxed);
    out.xruns = xruns_.load(std::memory_order_relaxed);
    out.ring_overruns = overruns_.load(std::memory_order_relaxed);
    out.ring_underruns = underruns_.load(std::memory_order_relaxed);
    out.callback_ns_min = ns_min_.load(std::memory_order_relaxed);
    out.callback_ns_max = ns_max_.load(std::memory_order_relaxed);
    out.callback_ns_total = ns_total_.load(std::memory_order_relaxed);
    out.period_ns = period_ns_.load(std::memory_order_relaxed);
    out.plan_generation = plan_generation_.load(std::memory_order_relaxed);
    out.rt_allocations = rt_allocations_.load(std::memory_order_relaxed);
    out.bucket_ns = bucket_ns_;
    for (int i = 0; i < TS_HISTOGRAM_BUCKETS; ++i) out.histogram[i] = histogram_[i].load(std::memory_order_relaxed);
}

void Engine::reset_stats() { stats_reset_generation_.fetch_add(1, std::memory_order_release); }

ts_result Engine::get_meter(uint32_t node_id, ts_meter& out) {
    if (!latest_) { fail("no plan applied"); return TS_ERR_STATE; }
    auto it = latest_->node_index.find(node_id);
    if (it == latest_->node_index.end()) { fail("no such node"); return TS_ERR_NOT_FOUND; }
    const Node& node = latest_->nodes[it->second];
    out.peak = node.meter.peak.load(std::memory_order_relaxed);
    out.rms = node.meter.rms.load(std::memory_order_relaxed);
    out.clipped = node.meter.clipped.load(std::memory_order_relaxed);
    out.channels = node.channels;
    return TS_OK;
}

void Engine::reset_meters() { meter_reset_generation_.fetch_add(1, std::memory_order_release); }

int32_t Engine::poll_events(ts_event* out, int32_t capacity) {
    int32_t n = 0;
    while (n < capacity && events_.pop(out[n])) ++n;
    return n;
}

}  // namespace ts
