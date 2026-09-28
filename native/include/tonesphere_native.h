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
#define TS_ABI_VERSION 3

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

#define TS_EQ_OFF        0
#define TS_EQ_PEAKING    1
#define TS_EQ_LOW_SHELF  2
#define TS_EQ_HIGH_SHELF 3
#define TS_EQ_HIGHPASS   4
#define TS_EQ_LOWPASS    5

#define TS_INSERT_FLAG_BYPASSED 0x1u

typedef struct ts_insert_desc {
    uint32_t node_id;
    uint32_t slot;   /* 0..TS_MAX_INSERTS-1, unique per node; also the processing order */
    uint32_t type;   /* TS_INSERT_* */
    uint32_t flags;  /* TS_INSERT_FLAG_* */
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
