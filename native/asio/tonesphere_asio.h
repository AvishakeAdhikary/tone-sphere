/*
 * ToneSphere ASIO host — C ABI.
 *
 * Copyright (C) 2026 Neural Nexus Studios
 *
 * This program is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software Foundation,
 * version 3. It is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 * PARTICULAR PURPOSE. See LICENSE in this directory.
 *
 * This library uses the Steinberg ASIO SDK under its GPLv3 licensing option, which is why
 * it is a separate DLL under GPLv3 while the rest of ToneSphere's source is MIT. It drives
 * tonesphere_native.dll's engine only through that library's C ABI.
 *
 * One ASIO driver per process: ASIO's callbacks carry no context pointer, so every ASIO
 * host has this limit. Starting a second stream while one runs is refused.
 */
#ifndef TONESPHERE_ASIO_H
#define TONESPHERE_ASIO_H

#include <stdint.h>

#include "tonesphere_native.h"

#ifdef TONESPHERE_ASIO_BUILD
#  define TS_ASIO_API __declspec(dllexport)
#else
#  define TS_ASIO_API __declspec(dllimport)
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define TS_ASIO_ABI_VERSION 1
#define TS_ASIO_MAX_CHANNELS 64

typedef struct ts_asio_driver {
    char name[128];        /* the registry key under HKLM\SOFTWARE\ASIO — what hosts show */
    char description[128];
    char clsid[64];
    char dll_path[260];    /* the driver's InprocServer32 */
    uint32_t dll_present;  /* 0: registered, but its DLL is missing — a broken install */
    uint32_t reserved;
} ts_asio_driver;

typedef struct ts_asio_channel {
    char name[32];
    int32_t sample_type;   /* ASIOSampleType */
    int32_t group;
    uint32_t supported;    /* ToneSphere can convert this sample type */
    uint32_t reserved;
} ts_asio_channel;

/* Everything the driver reports about itself, at its current settings. */
typedef struct ts_asio_info {
    char driver_name[128];
    int32_t driver_version;
    int32_t inputs;
    int32_t outputs;
    int32_t min_buffer;
    int32_t max_buffer;
    int32_t preferred_buffer;
    int32_t granularity;          /* -1: powers of two between min and max */
    double current_sample_rate;
    uint32_t rates_supported;     /* bit i: TS_ASIO_RATES[i] accepted by canSampleRate */
    int32_t input_latency;        /* frames, reported by the driver for the preferred buffer */
    int32_t output_latency;
    uint32_t post_output;         /* the driver supports outputReady() */
    ts_asio_channel input_channels[TS_ASIO_MAX_CHANNELS];
    ts_asio_channel output_channels[TS_ASIO_MAX_CHANNELS];
} ts_asio_info;

/* The rates probed for rates_supported, in bit order. */
#define TS_ASIO_RATE_COUNT 8
static const double TS_ASIO_RATES[TS_ASIO_RATE_COUNT] = {44100, 48000, 88200, 96000, 176400, 192000, 32000, 22050};

typedef struct ts_asio_config {
    char driver[128];            /* registry name, as ts_asio_list reports it */
    uint32_t buffer_frames;      /* 0 = the driver's preferred size */
    uint32_t input_node;         /* engine SOURCE node for the inputs; 0 = no inputs */
    uint32_t output_node;        /* engine SINK node for the outputs; 0 = no outputs */
    uint32_t input_count;
    uint32_t output_count;
    uint32_t inputs[TS_ASIO_MAX_CHANNELS];   /* driver channel indices, in node-channel order */
    uint32_t outputs[TS_ASIO_MAX_CHANNELS];
} ts_asio_config;

TS_ASIO_API int32_t ts_asio_abi_version(void);
TS_ASIO_API int32_t ts_asio_last_error(char* buffer, int32_t capacity);
TS_ASIO_API int32_t ts_asio_list(ts_asio_driver* out, int32_t capacity);
/* Loads the driver, reads what it reports, and unloads it. */
TS_ASIO_API ts_result ts_asio_query(const char* driver, ts_asio_info* out);
/* Loads and starts the driver at the engine's sample rate, and attaches it to the engine
 * as its backend. Stop with ts_engine_stop_backend; status through ts_engine_stream_status. */
TS_ASIO_API ts_result ts_asio_start(ts_engine* engine, const ts_asio_config* config);

/* Exposed so the boundary can be tested without a driver. */
TS_ASIO_API ts_result ts_asio_convert_in(int32_t sample_type, const void* src, float* dst, uint32_t samples);
TS_ASIO_API ts_result ts_asio_convert_out(int32_t sample_type, const float* src, void* dst, uint32_t samples);

#ifdef __cplusplus
}
#endif

#endif
