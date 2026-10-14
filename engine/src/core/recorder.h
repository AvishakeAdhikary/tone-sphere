#pragma once
// Multitrack recording and instant replay.
// The audio thread pushes post-fader audio of tapped nodes into lock-free rings
// (NodeRuntime::tap); a writer thread drains them to WAV files and/or keeps the last
// N seconds in memory for "save replay".
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "core/ring_buffer.h"

namespace ts {

struct NodeRuntime;

/// Writes 32-bit float WAV (RIFF, upgraded to RF64 when larger than 4 GB).
class WavWriter {
public:
    bool open(const std::string& path, int channels, double rate);
    void write(const float* interleaved, size_t frames);
    void close();
    bool isOpen() const { return f_ != nullptr; }
    const std::string& path() const { return path_; }

private:
    FILE* f_ = nullptr;
    std::string path_;
    int channels_ = 2;
    double rate_ = 48000;
    uint64_t dataBytes_ = 0;
};

class Recorder {
public:
    Recorder();
    ~Recorder();

    struct Tap {
        std::string nodeId, name;
        std::shared_ptr<NodeRuntime> rt;
        int channels = 2;
    };
    /// Sets the nodes that are tapped (recording and/or replay). Control thread.
    void setTaps(const std::vector<Tap>& taps, double rate);
    void setReplaySeconds(int seconds);
    int replaySeconds() const { return replaySeconds_.load(); }

    /// Starts writing one WAV per tapped node into `folder`. Returns false with `error` set.
    bool start(const std::vector<std::string>& nodeIds, const std::string& folder, std::string& error);
    std::vector<std::string> stop();
    bool recording() const { return recording_.load(); }
    double startedAt() const { return startedAt_; }
    /// Writes the replay buffer (last `seconds`) of every replay node; returns the files.
    std::vector<std::string> saveReplay(int seconds, const std::string& folder, std::string& error);

private:
    struct Track {
        Tap tap;
        std::unique_ptr<FrameRing> ring;      // audio thread -> writer
        std::unique_ptr<WavWriter> file;      // while recording this node
        std::vector<float> replay;            // circular, replaySeconds * rate frames
        size_t replayPos = 0, replayFill = 0;
    };
    void writerLoop();
    void detachAllLocked();

    std::mutex mu_;
    std::map<std::string, Track> tracks_;
    double rate_ = 48000;
    std::atomic<int> replaySeconds_{0};
    std::atomic<bool> recording_{false};
    double startedAt_ = 0;
    std::thread writer_;
    std::atomic<bool> running_{true};
};

}  // namespace ts
