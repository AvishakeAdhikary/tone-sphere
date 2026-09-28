// The engine: an immutable, preallocated execution plan per routing configuration, run by
// exactly one audio thread, replaced by the control thread without either side blocking.
//
// Plan exchange. The control thread builds a Plan completely — validated, topologically
// ordered, every buffer allocated — and publishes it with one atomic exchange. The audio
// thread announces the plan it is using through a single hazard pointer, and the control
// thread frees a retired plan only once the hazard no longer names it. The audio side is
// lock-free (it retries its load only if a publish races it, which cannot happen more than
// once per publish); the control side never waits — a plan still in use is simply freed
// on a later call.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "spsc.h"
#include "tonesphere_native.h"

namespace ts {

struct MeterState {
    std::atomic<float> peak{0.0f};
    std::atomic<float> rms{0.0f};
    std::atomic<uint32_t> clipped{0};
    // Audio-thread-owned.
    float peak_since_reset = 0.0f;
    uint32_t reset_seen = 0;
};

struct Node {
    uint32_t id = 0;
    uint32_t kind = 0;
    uint32_t channels = 0;
    uint32_t flags = 0;
    float* buffer = nullptr;  // planar, `channels` rows of `max_block` frames
    FrameRing* ring = nullptr;
    uint32_t route_begin = 0;  // incoming routes: Plan::routes[route_begin, route_end)
    uint32_t route_end = 0;
    MeterState meter;
    // Audio-thread-owned: so a starved or overflowing ring reports once when it starts,
    // not once per block for as long as it lasts.
    bool ring_trouble = false;
    bool nonfinite = false;
};

struct Route {
    uint32_t source_index = 0;
    uint64_t key = 0;
    bool invert = false;
    float pan_left = 1.0f;     // mono -> stereo, constant power
    float pan_right = 1.0f;
    float balance_left = 1.0f; // stereo -> stereo, unity at centre
    float balance_right = 1.0f;
    std::atomic<float> target{1.0f};
    std::atomic<uint32_t> muted{0};
    std::atomic<float> published{0.0f};  // last gain the audio thread reached; read when carrying over
    float current = 0.0f;                // audio-thread-owned
};

struct Plan {
    uint64_t generation = 0;
    uint32_t node_count = 0;
    uint32_t route_count = 0;
    std::unique_ptr<Node[]> nodes;    // topological order
    std::unique_ptr<Route[]> routes;  // grouped by destination, in node order
    std::unique_ptr<float[]> storage;
    std::unique_ptr<float[]> scratch;  // interleaved staging for ring I/O
    std::vector<std::shared_ptr<FrameRing>> rings;
    // Control-thread only.
    std::unordered_map<uint32_t, uint32_t> node_index;
    std::unordered_map<uint64_t, uint32_t> route_index;
};

inline uint64_t route_key(uint32_t source, uint32_t dest) {
    return (static_cast<uint64_t>(source) << 32) | dest;
}

class Engine {
public:
    Engine(uint32_t sample_rate, uint32_t max_block);
    ~Engine();

    ts_result apply_plan(const ts_node_desc* nodes, uint32_t node_count,
                         const ts_route_desc* routes, uint32_t route_count);
    ts_result set_route_gain(uint32_t source, uint32_t dest, float gain);
    ts_result set_route_muted(uint32_t source, uint32_t dest, bool muted);
    ts_result set_master_gain(float gain);

    ts_result process_offline(const ts_port_buffer* inputs, uint32_t input_count,
                              ts_port_buffer* outputs, uint32_t output_count, uint32_t frames);

    // The real-time entry point. Allocation-free, lock-free, never throws.
    void run_block(const ts_port_buffer* inputs, uint32_t input_count,
                   ts_port_buffer* outputs, uint32_t output_count, uint32_t frames) noexcept;

    int32_t port_write(uint32_t node_id, const float* data, uint32_t frames);
    int32_t port_read(uint32_t node_id, float* data, uint32_t frames);
    int32_t port_available(uint32_t node_id);

    void get_stats(ts_stats& out) const;
    void reset_stats();
    ts_result get_meter(uint32_t node_id, ts_meter& out);
    void reset_meters();
    int32_t poll_events(ts_event* out, int32_t capacity);

    void fail(const std::string& message) { last_error_ = message; }
    const std::string& last_error() const { return last_error_; }

    uint32_t sample_rate() const { return sample_rate_; }
    uint32_t max_block() const { return max_block_; }

private:
    Plan* acquire() noexcept;
    void release() noexcept;
    void collect();
    void process_plan(Plan& plan, const ts_port_buffer* inputs, uint32_t input_count,
                      ts_port_buffer* outputs, uint32_t output_count, uint32_t frames) noexcept;
    void fill_source(Plan& plan, Node& node, const ts_port_buffer* inputs, uint32_t input_count,
                     uint32_t frames) noexcept;
    void mix_into(Plan& plan, Node& node, uint32_t frames) noexcept;
    void emit_sink(Plan& plan, Node& node, ts_port_buffer* outputs, uint32_t output_count,
                   uint32_t frames, float master_from, float master_to) noexcept;
    void meter(Node& node, uint32_t frames, uint32_t reset_generation) noexcept;
    void post(uint32_t code, uint32_t arg0, uint64_t arg1) noexcept;
    FrameRing* find_ring(uint32_t node_id, uint32_t kind);

    const uint32_t sample_rate_;
    const uint32_t max_block_;
    std::string last_error_;

    std::atomic<Plan*> current_{nullptr};
    std::atomic<Plan*> hazard_{nullptr};
    Plan* latest_ = nullptr;  // control thread's view of current_
    std::vector<Plan*> retired_;
    std::unordered_map<uint32_t, std::shared_ptr<FrameRing>> rings_;
    uint64_t next_generation_ = 1;

    std::atomic<float> master_target_{1.0f};
    float master_current_ = 1.0f;  // audio-thread-owned

    std::atomic<uint32_t> meter_reset_generation_{1};
    std::atomic<uint32_t> stats_reset_generation_{1};
    uint32_t stats_reset_seen_ = 1;  // audio-thread-owned
    uint64_t rt_allocations_baseline_ = 0;

    // Written only by the audio thread; read by the control thread.
    std::atomic<uint64_t> blocks_{0};
    std::atomic<uint64_t> xruns_{0};
    std::atomic<uint64_t> overruns_{0};
    std::atomic<uint64_t> underruns_{0};
    std::atomic<uint64_t> ns_min_{UINT64_MAX};
    std::atomic<uint64_t> ns_max_{0};
    std::atomic<uint64_t> ns_total_{0};
    std::atomic<uint64_t> period_ns_{0};
    std::atomic<uint64_t> plan_generation_{0};
    std::atomic<uint64_t> rt_allocations_{0};
    std::atomic<uint64_t> histogram_[TS_HISTOGRAM_BUCKETS] = {};
    const uint64_t bucket_ns_;

    ItemQueue<ts_event, 256> events_;
    uint32_t events_lost_ = 0;  // audio-thread-owned

    std::atomic<bool> backend_running_{false};
};

}  // namespace ts
