// The exported C functions. Every control entry point converts C++ exceptions into a
// ts_result and a last_error message: an exception must never cross into Python's
// ctypes frame, where it would terminate the process.
#include <cstring>
#include <new>
#include <string>

#include "convert.h"
#include "engine.h"
#include "resampler.h"
#ifdef _WIN32
#include <mutex>

#include "../windows_audio/wasapi.h"
#endif

using ts::Engine;

struct ts_engine {
    Engine engine;
    ts_engine(uint32_t rate, uint32_t block) : engine(rate, block) {}
};

struct ts_ring {
    ts::FrameRing ring;
    ts_ring(uint32_t frames, uint32_t channels) : ring(frames, channels) {}
};

struct ts_resampler {
    ts::FrameRing ring;
    ts::DriftResampler resampler;
    ts_resampler(uint32_t channels, uint32_t max_block, uint32_t target, uint32_t ring_frames)
        : ring(ring_frames, channels), resampler(channels, max_block, target) {}
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

#ifdef _WIN32
namespace {
thread_local std::string t_wasapi_error;
std::mutex g_watcher_mutex;
std::unique_ptr<ts::wasapi::DeviceWatcher> g_watcher;
}  // namespace

TS_API int32_t ts_wasapi_enumerate(ts_device_info* out, int32_t capacity) {
    if (capacity < 0 || (capacity && !out)) return TS_ERR_INVALID;
    try {
        return ts::wasapi::enumerate(out, capacity, t_wasapi_error);
    } catch (const std::exception& e) {
        t_wasapi_error = e.what();
        return TS_ERR_BACKEND;
    }
}

TS_API int32_t ts_wasapi_last_error(char* buffer, int32_t capacity) {
    if (!buffer || capacity <= 0) return 0;
    const int32_t n = static_cast<int32_t>(t_wasapi_error.size()) < capacity - 1 ? static_cast<int32_t>(t_wasapi_error.size())
                                                                                 : capacity - 1;
    std::memcpy(buffer, t_wasapi_error.data(), static_cast<size_t>(n));
    buffer[n] = '\0';
    return n;
}

TS_API ts_result ts_wasapi_watch(int32_t enable) {
    std::lock_guard<std::mutex> lock(g_watcher_mutex);
    try {
        if (!enable) {
            g_watcher.reset();
            return TS_OK;
        }
        if (g_watcher) return TS_OK;
        auto watcher = std::make_unique<ts::wasapi::DeviceWatcher>();
        if (!watcher->start(t_wasapi_error)) return TS_ERR_BACKEND;
        g_watcher = std::move(watcher);
        return TS_OK;
    } catch (const std::exception& e) {
        t_wasapi_error = e.what();
        return TS_ERR_BACKEND;
    }
}

TS_API int32_t ts_wasapi_poll_events(ts_device_event* out, int32_t capacity) {
    if (!out || capacity < 0) return TS_ERR_INVALID;
    std::lock_guard<std::mutex> lock(g_watcher_mutex);
    if (!g_watcher) {
        t_wasapi_error = "not watching; call ts_wasapi_watch(1) first";
        return TS_ERR_STATE;
    }
    return g_watcher->poll(out, capacity);
}

TS_API ts_result ts_engine_start_wasapi(ts_engine* engine, const ts_stream_desc* streams, uint32_t count,
                                        uint32_t master_index) {
    return guarded(engine, [&](Engine& e) -> ts_result {
        if (e.backend_running()) {
            e.fail("a backend is already running; stop it first");
            return TS_ERR_STATE;
        }
        std::string error;
        auto backend = ts::wasapi::start(e, streams, count, master_index, error);
        if (!backend) {
            e.fail(error);
            return TS_ERR_BACKEND;
        }
        return e.attach_backend(std::move(backend));
    });
}
#endif

TS_API ts_result ts_engine_stop_backend(ts_engine* engine) {
    return guarded(engine, [&](Engine& e) { return e.stop_backend(); });
}

TS_API int32_t ts_engine_stream_status(ts_engine* engine, ts_stream_status* out, int32_t capacity) {
    if (!engine || !out || capacity < 0) return TS_ERR_INVALID;
    return engine->engine.backend_status(out, capacity);
}

namespace {
bool valid_format(uint32_t f) { return f >= TS_FORMAT_FLOAT32 && f <= TS_FORMAT_INT32; }
}  // namespace

TS_API ts_result ts_convert_to_float(uint32_t format, const void* src, float* dst, uint32_t samples) {
    if (!valid_format(format) || !src || !dst) return TS_ERR_INVALID;
    ts::to_float(static_cast<ts::SampleFormat>(format), src, dst, samples);
    return TS_OK;
}

TS_API ts_result ts_convert_from_float(uint32_t format, const float* src, void* dst, uint32_t samples) {
    if (!valid_format(format) || !src || !dst) return TS_ERR_INVALID;
    ts::from_float(static_cast<ts::SampleFormat>(format), src, dst, samples);
    return TS_OK;
}

TS_API ts_resampler* ts_resampler_create(uint32_t channels, uint32_t max_block, uint32_t target_fill,
                                         uint32_t ring_frames) {
    if (channels < 1 || channels > TS_MAX_CHANNELS || max_block < 1 || target_fill < 1 || ring_frames < target_fill)
        return nullptr;
    try {
        return new ts_resampler(channels, max_block, target_fill, ring_frames);
    } catch (...) {
        return nullptr;
    }
}

TS_API void ts_resampler_destroy(ts_resampler* r) { delete r; }
TS_API uint32_t ts_resampler_push(ts_resampler* r, const float* data, uint32_t frames) {
    return r && data ? r->ring.write(data, frames) : 0;
}
TS_API uint32_t ts_resampler_pull(ts_resampler* r, float* out, uint32_t frames) {
    return r && out ? r->resampler.process(r->ring, out, frames) : 0;
}
TS_API double ts_resampler_ratio(ts_resampler* r) { return r ? r->resampler.ratio() : 0.0; }
TS_API uint32_t ts_resampler_fill(ts_resampler* r) { return r ? r->ring.available() : 0; }

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
