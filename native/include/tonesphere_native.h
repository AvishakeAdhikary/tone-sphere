/*
 * ToneSphere native engine — the C ABI.
 *
 * This header is the whole boundary between the Python control plane and the native
 * real-time plane. It is deliberately flat C: plain structs, integer handles, no C++
 * types, no callbacks into the caller. Python binds it with ctypes
 * (tonesphere/native/_abi.py mirrors every struct here and must be kept in step).
 *
 * Threading contract:
 *   - Every ts_engine_* function except ts_engine_process is a CONTROL call. Control
 *     calls on one engine must be serialised by the caller (the Python wrapper holds a
 *     lock). They may allocate, and never run on the audio thread.
 *   - The audio thread is whichever thread runs the engine's blocks: a device backend's
 *     callback, or the caller of ts_engine_process when no backend is running. Exactly
 *     one such thread at a time.
 *   - ts_port_write / ts_port_read are the producer / consumer ends of a single-producer
 *     single-consumer ring. One thread per end.
 *
 * Every function returns ts_result (0 = success, negative = failure) unless noted;
 * ts_engine_last_error() then holds a human-readable reason.
 */
#ifndef TONESPHERE_NATIVE_H
#define TONESPHERE_NATIVE_H

#include <stdint.h>

#ifdef _WIN32
#  ifdef TONESPHERE_NATIVE_BUILD
#    define TS_API __declspec(dllexport)
#  else
#    define TS_API __declspec(dllimport)
#  endif
#else
#  define TS_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Bumped whenever a struct layout or a signature changes; Python refuses a mismatch. */
#define TS_ABI_VERSION 7

typedef int32_t ts_result;
#define TS_OK               0
#define TS_ERR_INVALID     -1  /* bad argument or malformed plan */
#define TS_ERR_CYCLE       -2  /* plan contains a feedback loop */
#define TS_ERR_NOMEM       -3
#define TS_ERR_STATE       -4  /* not allowed in the engine's current state */
#define TS_ERR_UNSUPPORTED -5
#define TS_ERR_NOT_FOUND   -6
#define TS_ERR_BACKEND     -7  /* the audio API refused; see last_error */

typedef struct ts_engine ts_engine;

/* ---- Plan ---------------------------------------------------------------------- */

#define TS_NODE_SOURCE 1  /* audio enters the graph here: a device input, a ring, or a process() input */
#define TS_NODE_BUS    2  /* a summing point; may feed further nodes */
#define TS_NODE_SINK   3  /* audio leaves the graph here: a device output, a ring, or a process() output */

/* The node's audio arrives from (SOURCE) or departs into (SINK) an SPSC ring that the
 * control plane writes or reads with ts_port_write / ts_port_read — network audio,
 * per-process capture, anything produced or consumed off the audio thread. */
#define TS_NODE_FLAG_RING 0x1u
/* SINK only: a sample-accurate safety limiter after master gain, so a routing mistake is
 * a compressed mix rather than full-scale noise. Adds no latency (see engine/dsp.h). */
#define TS_NODE_FLAG_LIMITER 0x2u

typedef struct ts_node_desc {
    uint32_t id;           /* caller-chosen, unique within the plan */
    uint32_t kind;         /* TS_NODE_* */
    uint32_t channels;     /* 1..TS_MAX_CHANNELS */
    uint32_t flags;        /* TS_NODE_FLAG_* */
    uint32_t ring_frames;  /* ring capacity when TS_NODE_FLAG_RING; rounded up to a power of two */
} ts_node_desc;

#define TS_ROUTE_FLAG_MUTED  0x1u
#define TS_ROUTE_FLAG_INVERT 0x2u

typedef struct ts_route_desc {
    uint32_t source;  /* node id: SOURCE or BUS */
    uint32_t dest;    /* node id: BUS or SINK */
    float gain;       /* linear */
    float pan;        /* -1 (left) .. +1 (right); applied when the destination is stereo */
    uint32_t flags;   /* TS_ROUTE_FLAG_* */
} ts_route_desc;

/* Built-in processors placed on a node, run in slot order after the node's input is
 * mixed and its trim/polarity applied, and before its fader. */
#define TS_INSERT_EQ         1  /* 8 bands x [type, frequency Hz, Q or shelf slope, gain dB]; type TS_EQ_* */
#define TS_INSERT_COMPRESSOR 2  /* [threshold dB, ratio, attack ms, release ms, knee dB, makeup dB] */
#define TS_INSERT_LIMITER    3  /* [threshold linear, release ms] */
#define TS_INSERT_DELAY      4  /* [delay ms, feedback, mix] */
#define TS_INSERT_VST3       5  /* a VST3 plugin opened with ts_vst3_open; `plugin` is its handle */

#define TS_EQ_OFF        0
#define TS_EQ_PEAKING    1
#define TS_EQ_LOW_SHELF  2
#define TS_EQ_HIGH_SHELF 3
#define TS_EQ_HIGHPASS   4
#define TS_EQ_LOWPASS    5

#define TS_INSERT_FLAG_BYPASSED 0x1u

typedef struct ts_insert_desc {
    uint32_t node_id;
    uint32_t slot;    /* 0..TS_MAX_INSERTS-1, unique per node; also the processing order */
    uint32_t type;    /* TS_INSERT_* */
    uint32_t flags;   /* TS_INSERT_FLAG_* */
    uint32_t plugin;  /* TS_INSERT_VST3 only: the ts_vst3_open handle; each at most once per plan */
} ts_insert_desc;

typedef struct ts_plan {
    const ts_node_desc* nodes;
    uint32_t node_count;
    const ts_route_desc* routes;
    uint32_t route_count;
    const ts_insert_desc* inserts;
    uint32_t insert_count;
} ts_plan;

#define TS_MAX_CHANNELS 64
#define TS_MAX_NODES    1024
#define TS_MAX_ROUTES   8192
#define TS_MAX_INSERTS  16

/* ---- Statistics, meters, events ----------------------------------------------------- */

/* Callback duration histogram, log-scaled so a 3 µs block and a 3 ms block are both
 * resolved: bucket i counts durations below 1000 * 2^((i + 1) / 4) ns (about 19% per
 * bucket, 1.19 µs to 65 ms); the last bucket also takes everything longer. */
#define TS_HISTOGRAM_BUCKETS 64

typedef struct ts_stats {
    uint64_t blocks;             /* blocks processed since the last reset */
    uint64_t xruns;              /* backend-reported dropouts */
    uint64_t ring_overruns;      /* frames dropped because a ring was full */
    uint64_t ring_underruns;     /* frames zero-filled because a ring was empty */
    uint64_t callback_ns_min;    /* UINT64_MAX when blocks == 0 */
    uint64_t callback_ns_max;
    uint64_t callback_ns_total;  /* mean = total / blocks */
    uint64_t period_ns;          /* the block period the last block was given (frames / rate) */
    uint64_t plan_generation;    /* generation of the plan the audio thread last ran */
    uint64_t rt_allocations;     /* heap allocations made by this DLL on the audio thread */
    uint64_t histogram[TS_HISTOGRAM_BUCKETS];
} ts_stats;

typedef struct ts_meter {
    float peak;          /* max |x| since the last meter reset, all channels */
    float rms;           /* RMS of the most recent block */
    uint32_t clipped;    /* 1 if any sample reached |x| >= 1.0 since the last reset */
    uint32_t channels;
} ts_meter;

/* Events the audio thread reports without logging. Drained by ts_engine_poll_events. */
#define TS_EVENT_RING_OVERRUN   1  /* arg0 = node id, arg1 = frames dropped */
#define TS_EVENT_RING_UNDERRUN  2  /* arg0 = node id, arg1 = frames zero-filled */
#define TS_EVENT_NONFINITE      3  /* arg0 = node id; a NaN/inf was replaced with silence */
#define TS_EVENT_XRUN           4  /* arg0 = backend-specific flags */
#define TS_EVENT_EVENTS_LOST    5  /* arg0 = number of events dropped because the event ring was full */

typedef struct ts_event {
    uint32_t code;
    uint32_t arg0;
    uint64_t arg1;
    uint64_t block;  /* block counter at the time */
} ts_event;

/* ---- Offline processing --------------------------------------------------------------- */

/* Interleaved float32, `frames` x `channels`, row stride = channels. */
typedef struct ts_port_buffer {
    uint32_t node_id;
    uint32_t channels;
    float* data;
} ts_port_buffer;

/* ---- Functions ------------------------------------------------------------------------ */

TS_API int32_t ts_abi_version(void);
TS_API const char* ts_build_info(void);

TS_API ts_engine* ts_engine_create(uint32_t sample_rate, uint32_t max_block_frames);
TS_API void ts_engine_destroy(ts_engine* engine);

/* Copies the last failure's reason into `buffer` (always NUL-terminated). Returns its length. */
TS_API int32_t ts_engine_last_error(ts_engine* engine, char* buffer, int32_t capacity);

/* Validate, order and preallocate a plan, then publish it to the audio thread. On failure
 * the running plan is untouched. State that belongs to something the new plan still has —
 * a route's smoothed gain, a node's fader/trim/polarity, an insert's parameters and filter
 * memory, a ring's queued audio — carries over, so a swap neither clicks nor forgets. */
TS_API ts_result ts_engine_apply_plan(ts_engine* engine, const ts_plan* plan);

/* Continuous controls — atomic, take effect on the next block, ramped over one block. */
TS_API ts_result ts_engine_set_route_gain(ts_engine* engine, uint32_t source, uint32_t dest, float gain);
TS_API ts_result ts_engine_set_route_muted(ts_engine* engine, uint32_t source, uint32_t dest, int32_t muted);
TS_API ts_result ts_engine_set_route_pan(ts_engine* engine, uint32_t source, uint32_t dest, float pan);
TS_API ts_result ts_engine_set_master_gain(ts_engine* engine, float gain);
TS_API ts_result ts_engine_set_node_gain(ts_engine* engine, uint32_t node_id, float gain);
TS_API ts_result ts_engine_set_node_muted(ts_engine* engine, uint32_t node_id, int32_t muted);
TS_API ts_result ts_engine_set_channel_trim(ts_engine* engine, uint32_t node_id, uint32_t channel, float gain);
TS_API ts_result ts_engine_set_channel_inverted(ts_engine* engine, uint32_t node_id, uint32_t channel, int32_t inverted);

TS_API ts_result ts_engine_set_insert_param(ts_engine* engine, uint32_t node_id, uint32_t slot,
                                            uint32_t param, float value);
TS_API ts_result ts_engine_get_insert_param(ts_engine* engine, uint32_t node_id, uint32_t slot,
                                            uint32_t param, float* value);
TS_API ts_result ts_engine_set_insert_bypassed(ts_engine* engine, uint32_t node_id, uint32_t slot, int32_t bypassed);
/* Gain reduction in dB the audio thread last applied (compressor, limiter), or NaN. */
TS_API ts_result ts_engine_get_insert_readout(ts_engine* engine, uint32_t node_id, uint32_t slot, float* value);

/* Run one block on the calling thread with no device. Inputs feed SOURCE nodes that are
 * not rings; outputs receive SINK nodes that are not rings. A source with no buffer
 * given is silent. Refused while a device backend is running. */
TS_API ts_result ts_engine_process(ts_engine* engine,
                                   const ts_port_buffer* inputs, uint32_t input_count,
                                   ts_port_buffer* outputs, uint32_t output_count,
                                   uint32_t frames);

/* Ring ports: producer / consumer ends for TS_NODE_FLAG_RING nodes. Interleaved float32.
 * Return the number of frames actually written / read (short on full / empty), or a
 * negative ts_result. */
TS_API int32_t ts_port_write(ts_engine* engine, uint32_t node_id, const float* data, uint32_t frames);
TS_API int32_t ts_port_read(ts_engine* engine, uint32_t node_id, float* data, uint32_t frames);
TS_API int32_t ts_port_available(ts_engine* engine, uint32_t node_id);

TS_API ts_result ts_engine_get_stats(ts_engine* engine, ts_stats* out);
TS_API ts_result ts_engine_reset_stats(ts_engine* engine);
TS_API ts_result ts_engine_get_meter(ts_engine* engine, uint32_t node_id, ts_meter* out);
TS_API ts_result ts_engine_reset_meters(ts_engine* engine);
TS_API int32_t ts_engine_poll_events(ts_engine* engine, ts_event* out, int32_t capacity);

/* ---- Windows audio (WASAPI) ------------------------------------------------------------- */

#define TS_DEVICE_ID_CHARS   256
#define TS_DEVICE_NAME_CHARS 256

#define TS_FLOW_RENDER  1
#define TS_FLOW_CAPTURE 2

#define TS_ROLE_CONSOLE        0x1u
#define TS_ROLE_MULTIMEDIA     0x2u
#define TS_ROLE_COMMUNICATIONS 0x4u

/* Strings are UTF-16 (wchar_t on Windows), NUL-terminated. The id is the MMDevice endpoint
 * ID: stable across reboots and re-enumeration, unlike a PortAudio index. */
typedef struct ts_device_info {
    uint16_t id[TS_DEVICE_ID_CHARS];
    uint16_t name[TS_DEVICE_NAME_CHARS];
    uint32_t flow;             /* TS_FLOW_* */
    uint32_t state;            /* DEVICE_STATE_* as Windows reports it (1 = active) */
    uint32_t default_roles;    /* TS_ROLE_* for which this is the default endpoint */
    uint32_t mix_channels;     /* the shared-mode mix format */
    uint32_t mix_sample_rate;
    uint32_t mix_bits;
    uint32_t mix_is_float;
    int64_t default_period_hns;
    int64_t min_period_hns;    /* the smallest exclusive-mode period */
    /* IAudioClient3 shared-mode engine periods in frames; 0 where Windows does not offer them */
    uint32_t shared_default_period_frames;
    uint32_t shared_fundamental_period_frames;
    uint32_t shared_min_period_frames;
    uint32_t shared_max_period_frames;
    uint32_t raw_supported;    /* the endpoint can bypass its enhancement effects (raw mode) */
    uint32_t reserved;
} ts_device_info;

#define TS_DEVICE_EVENT_ADDED           1
#define TS_DEVICE_EVENT_REMOVED         2
#define TS_DEVICE_EVENT_STATE_CHANGED   3
#define TS_DEVICE_EVENT_DEFAULT_CHANGED 4
#define TS_DEVICE_EVENT_LOST            5  /* state = number of events dropped because the queue was full */

typedef struct ts_device_event {
    uint32_t kind;
    uint32_t flow;   /* DEFAULT_CHANGED only */
    uint32_t role;   /* DEFAULT_CHANGED only: TS_ROLE_* */
    uint32_t state;  /* STATE_CHANGED: the new DEVICE_STATE_* */
    uint16_t id[TS_DEVICE_ID_CHARS];
} ts_device_event;

#define TS_STREAM_RENDER           1  /* play into an endpoint */
#define TS_STREAM_CAPTURE          2  /* record from an endpoint */
#define TS_STREAM_LOOPBACK         3  /* record what a render endpoint is playing (whole system) */
#define TS_STREAM_PROCESS_LOOPBACK 4  /* record what one process (and optionally its children) plays */

#define TS_SHARE_SHARED    0
#define TS_SHARE_EXCLUSIVE 1

/* Exclusive refused -> open shared instead, and say so in ts_stream_status. Without this
 * flag a refused exclusive request fails the stream. */
#define TS_STREAM_FLAG_ALLOW_SHARED_FALLBACK 0x1u
/* PROCESS_LOOPBACK: capture the process tree rather than exclude it. */
#define TS_STREAM_FLAG_INCLUDE_TREE          0x2u
/* Shared RENDER/CAPTURE: ask for raw stream processing, bypassing the driver's
 * enhancement effects (loudness equalisation, noise suppression, AGC). Honoured only
 * where the endpoint supports raw mode; ts_stream_status.raw says whether it does. */
#define TS_STREAM_FLAG_RAW                   0x4u

typedef struct ts_stream_desc {
    uint16_t device_id[TS_DEVICE_ID_CHARS];  /* empty for PROCESS_LOOPBACK */
    uint32_t node_id;      /* SOURCE node for capture kinds, SINK node for RENDER */
    uint32_t kind;         /* TS_STREAM_* */
    uint32_t share_mode;   /* TS_SHARE_* (RENDER and CAPTURE only) */
    uint32_t flags;        /* TS_STREAM_FLAG_* */
    uint32_t channels;     /* must equal the node's channel count */
    uint32_t process_id;   /* PROCESS_LOOPBACK only */
} ts_stream_desc;

#define TS_STREAM_STATE_STARTING 0
#define TS_STREAM_STATE_RUNNING  1
#define TS_STREAM_STATE_FAILED   2
#define TS_STREAM_STATE_STOPPED  3

typedef struct ts_stream_status {
    uint32_t node_id;
    uint32_t kind;
    uint32_t state;              /* TS_STREAM_STATE_* */
    uint32_t is_master;          /* this stream's device clock drives the engine */
    uint32_t share_mode;         /* what was actually obtained */
    int32_t hresult;             /* the failing HRESULT, or 0 */
    uint32_t sample_rate;        /* negotiated format */
    uint32_t channels;
    uint32_t bits;
    uint32_t valid_bits;
    uint32_t is_float;
    uint32_t buffer_frames;      /* the endpoint buffer WASAPI allocated */
    uint32_t period_frames;      /* the device period the stream wakes on */
    int64_t stream_latency_hns;  /* IAudioClient::GetStreamLatency: reported by Windows, not measured */
    uint64_t frames;             /* frames moved to or from the device */
    uint64_t glitches;           /* render: buffer found empty; capture: data discontinuity */
    uint64_t underruns;          /* frames invented as silence at the clock boundary */
    uint64_t overruns;           /* frames dropped at the clock boundary */
    double drift_ratio;          /* consumption ratio across the clock boundary; 1.0 for the master */
    uint32_t ring_fill;
    /* 1: raw processing requested and the endpoint reports raw support; 0: effects may be
     * applied (not requested, refused, or the endpoint does not support raw); exclusive
     * mode bypasses effects regardless. */
    uint32_t raw;
    char error[256];
} ts_stream_status;

TS_API int32_t ts_wasapi_enumerate(ts_device_info* out, int32_t capacity);
/* The reason the last ts_wasapi_* call on this thread failed. */
TS_API int32_t ts_wasapi_last_error(char* buffer, int32_t capacity);
TS_API ts_result ts_wasapi_watch(int32_t enable);
TS_API int32_t ts_wasapi_poll_events(ts_device_event* out, int32_t capacity);

/* Open the streams and run the engine from the master stream's device thread. Streams
 * other than the master cross a clock boundary through a ring with drift correction.
 * Fails if the master cannot open; a satellite that cannot open is reported FAILED in
 * ts_engine_stream_status and the rest run: partial success is visible, not hidden. */
TS_API ts_result ts_engine_start_wasapi(ts_engine* engine, const ts_stream_desc* streams, uint32_t count,
                                        uint32_t master_index);
TS_API ts_result ts_engine_stop_backend(ts_engine* engine);
TS_API int32_t ts_engine_stream_status(ts_engine* engine, ts_stream_status* out, int32_t capacity);

/* ---- VST3 plugins -------------------------------------------------------------------------
 *
 * Every call here runs on one plugin thread (OLE-initialised, with a message loop for
 * editors), because VST3 controllers expect a single UI thread. Calls into plugin code are
 * wrapped in structured exception handling: a plugin that faults while loading, opening,
 * closing or processing is marked crashed and never called again, and the host keeps
 * running. That catches access violations and C++ exceptions; a plugin that corrupts
 * memory before faulting can still take the process down, which in-process hosting cannot
 * prevent. */

typedef struct ts_vst3_class {
    char uid[64];            /* the class ID, 32 hex digits */
    char name[128];
    char vendor[128];
    char version[64];
    char category[64];       /* "Audio Module Class" for an audio processor */
    char subcategories[128]; /* e.g. "Fx|Delay" */
    char sdk_version[64];
    uint32_t class_flags;
    uint32_t is_audio_effect;
} ts_vst3_class;

typedef struct ts_vst3_param {
    uint32_t id;
    int32_t step_count;         /* 0 continuous, 1 toggle, n discrete */
    double default_normalized;
    double normalized;          /* the controller's current value */
    double plain;
    int32_t flags;              /* ParameterInfo::ParameterFlags: 1 can automate, 2 read-only, ... */
    int32_t unit_id;
    char title[128];
    char short_title[64];
    char units[32];
    char display[64];           /* the controller's own text for the current value */
} ts_vst3_param;

typedef struct ts_vst3_status {
    uint32_t crashed;           /* faulted once: bypassed from then on, never called again */
    uint32_t restart_flags;     /* IComponentHandler::restartComponent flags since the last status read */
    uint32_t latency;           /* samples, reported by the plugin (getLatencySamples) */
    uint32_t channels;
    uint64_t blocks;            /* process() calls */
    char fault[160];
} ts_vst3_status;

TS_API int32_t ts_vst3_last_error(char* buffer, int32_t capacity);
/* Load a module and list its classes, then unload it. Run this in a subprocess when the
 * plugin is untrusted: a crash in a plugin's module-load code cannot be contained in-process. */
TS_API int32_t ts_vst3_scan(const char* path, ts_vst3_class* out, int32_t capacity);
TS_API ts_result ts_vst3_open(const char* path, const char* class_uid, uint32_t sample_rate, uint32_t max_block,
                              uint32_t channels, uint32_t* handle);
/* The plugin is released once no running plan uses it any more. */
TS_API ts_result ts_vst3_close(uint32_t handle);
TS_API int32_t ts_vst3_param_count(uint32_t handle);
TS_API ts_result ts_vst3_param_info(uint32_t handle, int32_t index, ts_vst3_param* out);
/* Updates the controller and queues the change for the audio thread's next block. */
TS_API ts_result ts_vst3_set_param(uint32_t handle, uint32_t id, double normalized);
/* which: 0 component state, 1 controller state. Returns the size; copies up to capacity. */
TS_API int32_t ts_vst3_get_state(uint32_t handle, int32_t which, uint8_t* buffer, int32_t capacity);
TS_API ts_result ts_vst3_set_state(uint32_t handle, const uint8_t* component, int32_t component_size,
                                   const uint8_t* controller, int32_t controller_size);
TS_API ts_result ts_vst3_get_status(uint32_t handle, ts_vst3_status* out);
TS_API int32_t ts_vst3_has_editor(uint32_t handle);
TS_API ts_result ts_vst3_open_editor(uint32_t handle);
TS_API ts_result ts_vst3_close_editor(uint32_t handle);

/* ---- External device backends ------------------------------------------------------------
 *
 * A backend that lives in another DLL (the ASIO host, tonesphere_asio.dll, which is GPLv3
 * and therefore kept out of this MIT library) attaches through these. It owns the audio
 * thread while attached and calls ts_engine_run_block from it. The engine owns the ops
 * once attached: ts_engine_stop_backend calls stop() then destroy(). */

typedef struct ts_backend_ops {
    void* context;
    void (*stop)(void* context);
    int32_t (*status)(void* context, ts_stream_status* out, int32_t capacity);
    void (*destroy)(void* context);
} ts_backend_ops;

TS_API ts_result ts_engine_attach_backend(ts_engine* engine, const ts_backend_ops* ops);
/* AUDIO THREAD ONLY, and only by an attached backend. Never blocks, allocates or throws. */
TS_API void ts_engine_run_block(ts_engine* engine, const ts_port_buffer* inputs, uint32_t input_count,
                                ts_port_buffer* outputs, uint32_t output_count, uint32_t frames);
TS_API void ts_engine_set_backend_running(ts_engine* engine, int32_t running);
/* Any thread. */
TS_API void ts_engine_add_xruns(ts_engine* engine, uint32_t count);
TS_API uint32_t ts_engine_sample_rate(ts_engine* engine);
TS_API uint32_t ts_engine_max_block(ts_engine* engine);

/* ---- Pieces of the device boundary, exposed so they can be tested without a device ----- */

#define TS_FORMAT_FLOAT32 1
#define TS_FORMAT_FLOAT64 2
#define TS_FORMAT_INT16   3
#define TS_FORMAT_INT24   4  /* packed */
#define TS_FORMAT_INT32   5

TS_API ts_result ts_convert_to_float(uint32_t format, const void* src, float* dst, uint32_t samples);
TS_API ts_result ts_convert_from_float(uint32_t format, const float* src, void* dst, uint32_t samples);

typedef struct ts_resampler ts_resampler;
TS_API ts_resampler* ts_resampler_create(uint32_t channels, uint32_t max_block, uint32_t target_fill,
                                         uint32_t ring_frames);
TS_API void ts_resampler_destroy(ts_resampler* resampler);
TS_API uint32_t ts_resampler_push(ts_resampler* resampler, const float* data, uint32_t frames);
/* Returns frames that had to be invented as silence. */
TS_API uint32_t ts_resampler_pull(ts_resampler* resampler, float* out, uint32_t frames);
TS_API double ts_resampler_ratio(ts_resampler* resampler);
TS_API uint32_t ts_resampler_fill(ts_resampler* resampler);

/* ---- Standalone SPSC ring, for tests of the ring itself ---------------------------------- */

typedef struct ts_ring ts_ring;
TS_API ts_ring* ts_ring_create(uint32_t capacity_frames, uint32_t channels);
TS_API void ts_ring_destroy(ts_ring* ring);
TS_API uint32_t ts_ring_capacity(ts_ring* ring);
TS_API uint32_t ts_ring_write(ts_ring* ring, const float* data, uint32_t frames);
TS_API uint32_t ts_ring_read(ts_ring* ring, float* data, uint32_t frames);
TS_API uint32_t ts_ring_available(ts_ring* ring);

#ifdef __cplusplus
}
#endif

#endif
