// The exported C functions. Every control entry point converts C++ exceptions into a
// ts_result and a last_error message: an exception must never cross into Python's
// ctypes frame, where it would terminate the process.
#include <cstring>
#include <new>
#include <string>

#include "engine.h"

using ts::Engine;

struct ts_engine {
    Engine engine;
    ts_engine(uint32_t rate, uint32_t block) : engine(rate, block) {}
};

struct ts_ring {
    ts::FrameRing ring;
    ts_ring(uint32_t frames, uint32_t channels) : ring(frames, channels) {}
};

namespace {

template <typename F>
ts_result guarded(ts_engine* e, F&& body) {
    if (!e) return TS_ERR_INVALID;
    try {
        return body(e->engine);
    } catch (const std::bad_alloc&) {
        e->engine.fail("out of memory");
        return TS_ERR_NOMEM;
    } catch (const std::exception& ex) {
        e->engine.fail(ex.what());
        return TS_ERR_INVALID;
    }
}

}  // namespace

extern "C" {

TS_API int32_t ts_abi_version(void) { return TS_ABI_VERSION; }

TS_API const char* ts_build_info(void) {
    static const std::string info = [] {
        std::string s = "tonesphere_native abi=" + std::to_string(TS_ABI_VERSION);
#if defined(_MSC_FULL_VER)
        s += " msvc=" + std::to_string(_MSC_FULL_VER);
#elif defined(__clang__)
        s += " clang=" __clang_version__;
#endif
#ifdef NDEBUG
        s += " release";
#else
        s += " debug";
#endif
        return s;
    }();
    return info.c_str();
}

TS_API ts_engine* ts_engine_create(uint32_t sample_rate, uint32_t max_block_frames) {
    if (sample_rate < 8000 || sample_rate > 768000 || max_block_frames < 1 || max_block_frames > 8192) return nullptr;
    try {
        return new ts_engine(sample_rate, max_block_frames);
    } catch (...) {
        return nullptr;
    }
}

TS_API void ts_engine_destroy(ts_engine* engine) { delete engine; }

TS_API int32_t ts_engine_last_error(ts_engine* engine, char* buffer, int32_t capacity) {
    if (!engine || !buffer || capacity <= 0) return 0;
    const std::string& message = engine->engine.last_error();
    const int32_t n = static_cast<int32_t>(message.size()) < capacity - 1 ? static_cast<int32_t>(message.size()) : capacity - 1;
    std::memcpy(buffer, message.data(), static_cast<size_t>(n));
    buffer[n] = '\0';
    return n;
}

TS_API ts_result ts_engine_apply_plan(ts_engine* engine, const ts_plan* plan) {
    if (!plan) return TS_ERR_INVALID;
    return guarded(engine, [&](Engine& e) { return e.apply_plan(*plan); });
}

TS_API ts_result ts_engine_set_route_gain(ts_engine* engine, uint32_t source, uint32_t dest, float gain) {
    return guarded(engine, [&](Engine& e) { return e.set_route_gain(source, dest, gain); });
}

TS_API ts_result ts_engine_set_route_muted(ts_engine* engine, uint32_t source, uint32_t dest, int32_t muted) {
    return guarded(engine, [&](Engine& e) { return e.set_route_muted(source, dest, muted != 0); });
}

TS_API ts_result ts_engine_set_master_gain(ts_engine* engine, float gain) {
    return guarded(engine, [&](Engine& e) { return e.set_master_gain(gain); });
}

TS_API ts_result ts_engine_set_route_pan(ts_engine* engine, uint32_t source, uint32_t dest, float pan) {
    return guarded(engine, [&](Engine& e) { return e.set_route_pan(source, dest, pan); });
}

TS_API ts_result ts_engine_set_node_gain(ts_engine* engine, uint32_t node_id, float gain) {
    return guarded(engine, [&](Engine& e) { return e.set_node_gain(node_id, gain); });
}

TS_API ts_result ts_engine_set_node_muted(ts_engine* engine, uint32_t node_id, int32_t muted) {
    return guarded(engine, [&](Engine& e) { return e.set_node_muted(node_id, muted != 0); });
}

TS_API ts_result ts_engine_set_channel_trim(ts_engine* engine, uint32_t node_id, uint32_t channel, float gain) {
    return guarded(engine, [&](Engine& e) { return e.set_channel_trim(node_id, channel, gain); });
}

TS_API ts_result ts_engine_set_channel_inverted(ts_engine* engine, uint32_t node_id, uint32_t channel, int32_t inverted) {
    return guarded(engine, [&](Engine& e) { return e.set_channel_inverted(node_id, channel, inverted != 0); });
}

TS_API ts_result ts_engine_set_insert_param(ts_engine* engine, uint32_t node_id, uint32_t slot, uint32_t param,
                                            float value) {
    return guarded(engine, [&](Engine& e) { return e.set_insert_param(node_id, slot, param, value); });
}

TS_API ts_result ts_engine_get_insert_param(ts_engine* engine, uint32_t node_id, uint32_t slot, uint32_t param,
                                            float* value) {
    if (!value) return TS_ERR_INVALID;
    return guarded(engine, [&](Engine& e) { return e.get_insert_param(node_id, slot, param, *value); });
}

TS_API ts_result ts_engine_set_insert_bypassed(ts_engine* engine, uint32_t node_id, uint32_t slot, int32_t bypassed) {
    return guarded(engine, [&](Engine& e) { return e.set_insert_bypassed(node_id, slot, bypassed != 0); });
}

TS_API ts_result ts_engine_get_insert_readout(ts_engine* engine, uint32_t node_id, uint32_t slot, float* value) {
    if (!value) return TS_ERR_INVALID;
    return guarded(engine, [&](Engine& e) { return e.get_insert_readout(node_id, slot, *value); });
}

TS_API ts_result ts_engine_process(ts_engine* engine, const ts_port_buffer* inputs, uint32_t input_count,
                                   ts_port_buffer* outputs, uint32_t output_count, uint32_t frames) {
    if (!engine) return TS_ERR_INVALID;
    if ((input_count && !inputs) || (output_count && !outputs)) {
        engine->engine.fail("null port array");
        return TS_ERR_INVALID;
    }
    return engine->engine.process_offline(inputs, input_count, outputs, output_count, frames);
}

TS_API int32_t ts_port_write(ts_engine* engine, uint32_t node_id, const float* data, uint32_t frames) {
    if (!engine || (!data && frames)) return TS_ERR_INVALID;
    return engine->engine.port_write(node_id, data, frames);
}

TS_API int32_t ts_port_read(ts_engine* engine, uint32_t node_id, float* data, uint32_t frames) {
    if (!engine || (!data && frames)) return TS_ERR_INVALID;
    return engine->engine.port_read(node_id, data, frames);
}

TS_API int32_t ts_port_available(ts_engine* engine, uint32_t node_id) {
    if (!engine) return TS_ERR_INVALID;
    return engine->engine.port_available(node_id);
}

TS_API ts_result ts_engine_get_stats(ts_engine* engine, ts_stats* out) {
    if (!engine || !out) return TS_ERR_INVALID;
    engine->engine.get_stats(*out);
    return TS_OK;
}

TS_API ts_result ts_engine_reset_stats(ts_engine* engine) {
    if (!engine) return TS_ERR_INVALID;
    engine->engine.reset_stats();
    return TS_OK;
}

TS_API ts_result ts_engine_get_meter(ts_engine* engine, uint32_t node_id, ts_meter* out) {
    if (!out) return TS_ERR_INVALID;
    return guarded(engine, [&](Engine& e) { return e.get_meter(node_id, *out); });
}

TS_API ts_result ts_engine_reset_meters(ts_engine* engine) {
    if (!engine) return TS_ERR_INVALID;
    engine->engine.reset_meters();
    return TS_OK;
}

TS_API int32_t ts_engine_poll_events(ts_engine* engine, ts_event* out, int32_t capacity) {
    if (!engine || !out || capacity < 0) return TS_ERR_INVALID;
    return engine->engine.poll_events(out, capacity);
}

TS_API ts_ring* ts_ring_create(uint32_t capacity_frames, uint32_t channels) {
    if (capacity_frames < 1 || capacity_frames > (1u << 24) || channels < 1 || channels > TS_MAX_CHANNELS) return nullptr;
    try {
        return new ts_ring(capacity_frames, channels);
    } catch (...) {
        return nullptr;
    }
}

TS_API void ts_ring_destroy(ts_ring* ring) { delete ring; }
TS_API uint32_t ts_ring_capacity(ts_ring* ring) { return ring ? ring->ring.capacity() : 0; }
TS_API uint32_t ts_ring_write(ts_ring* ring, const float* data, uint32_t frames) {
    return ring && data ? ring->ring.write(data, frames) : 0;
}
TS_API uint32_t ts_ring_read(ts_ring* ring, float* data, uint32_t frames) {
    return ring && data ? ring->ring.read(data, frames) : 0;
}
TS_API uint32_t ts_ring_available(ts_ring* ring) { return ring ? ring->ring.available() : 0; }

}  // extern "C"
