#pragma once
// Audio graph runtime.
//
// The control thread owns the editable model (Engine). Whenever the topology changes it
// compiles an immutable Program (processing order, routes, crosspoints, FX lists) and
// publishes it atomically. The audio thread only ever reads the current Program plus
// per-node/per-route atomics, so it never blocks on the control thread.
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/bridge.h"
#include "dsp/fx.h"

namespace nt {

constexpr int kMaxChannels = 64;
constexpr int kMaxBlock = 2048;

enum class Bin { PhysicalIn, PhysicalOut, VirtualIn, VirtualOut, AppIn, AppOut, Bus };
const char* binName(Bin b);
bool binIsSource(Bin b);
bool binIsDest(Bin b);

/// Shared meter values (peak, rms per channel) read by the IPC thread at ~30 Hz.
struct MeterBank {
    explicit MeterBank(size_t count, uint32_t layout) : size(count), layoutVersion(layout), values(new std::atomic<float>[count ? count : 1]) {
        for (size_t i = 0; i < size; ++i) values[i].store(0.0f);
    }
    size_t size;
    uint32_t layoutVersion;
    std::unique_ptr<std::atomic<float>[]> values;
};

/// Simple test-tone / signal generator source.
struct ToneGen {
    std::atomic<bool> enabled{false};
    std::atomic<float> freq{440.0f};
    std::atomic<float> levelDb{-18.0f};
    double phase = 0.0;
};

/// Per-node state shared between the control and audio threads.
struct NodeRuntime {
    std::string id;
    Bin bin = Bin::PhysicalIn;
    int channels = 2;

    // Live parameters (control writes, audio reads).
    std::atomic<float> gainDb{0.0f};
    std::atomic<bool> muted{false};
    std::atomic<bool> solo{false};
    std::atomic<bool> invert{false};
    std::atomic<float> pan{0.0f};
    std::atomic<float> delayMs{0.0f};

    // Endpoints. Raw atomics read by the audio thread; owned (shared_ptr) by the engine model and
    // only destroyed after an RCU grace period, so swapping them never blocks audio.
    std::atomic<CaptureBridge*> capture{nullptr};  // sources backed by a device/app/network stream
    std::atomic<RenderBridge*> render{nullptr};    // destinations backed by a device/app/network stream
    std::shared_ptr<ToneGen> tone;
    // Recording / replay tap (owned by the Recorder; null when not tapped).
    std::atomic<FrameRing*> tap{nullptr};

    // Statistics (audio writes, control reads).
    std::atomic<float> cpuUs{0.0f};

    // ---- audio-thread state ----
    std::vector<float> storage;  // planar [channels][kMaxBlock]
    float* buf[kMaxChannels] = {};
    float curGain = 0.0f;  // fade in when a node first appears
    std::vector<float> delayLine;
    int delayCap = 0, delayPos = 0;
    std::vector<float> tapScratch;  // interleaved copy for the tap

    void allocate(int ch, double sampleRate);
};

struct RouteRuntime {
    std::string id;
    std::atomic<float> gainDb{0.0f};
    std::atomic<bool> muted{false};
    std::atomic<bool> enabled{true};
    float curGain = 0.0f;  // audio thread; starts silent so new connections fade in
};

struct Crosspoint {
    int from = 0, to = 0;
    float gain = 1.0f;  // linear (static part)
};

struct Incoming {
    NodeRuntime* from = nullptr;
    RouteRuntime* route = nullptr;
    std::vector<Crosspoint> xps;
};

struct ProgramNode {
    NodeRuntime* rt = nullptr;
    bool isSource = false;  // pulls from capture/tone
    bool isBus = false;
    bool isDest = false;     // pushes to render
    bool physicalOut = false;
    std::vector<Incoming> in;
    std::vector<std::shared_ptr<Fx>> fx;
    int meterOffset = 0;
};

struct Program {
    std::vector<ProgramNode> nodes;  // order: sources, buses (topological), destinations
    std::vector<std::shared_ptr<NodeRuntime>> keepNodes;
    std::vector<std::shared_ptr<RouteRuntime>> keepRoutes;
    std::shared_ptr<MeterBank> meters;
    std::vector<NodeRuntime*> soloBuses;
    double sampleRate = 48000.0;
};

/// Runs one block (frames <= kMaxBlock) of the program. Audio thread only.
void processProgram(Program& p, int frames, double nowSec);

}  // namespace nt
