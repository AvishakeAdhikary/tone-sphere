#include "core/recorder.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>

#include "core/graph.h"
#include "util/log.h"
#include "util/platform.h"

namespace nt {

namespace fs = std::filesystem;

// ------------------------------------------------------------------ WAV writer
namespace {
void put32(FILE* f, uint32_t v) { std::fwrite(&v, 4, 1, f); }
void put16(FILE* f, uint16_t v) { std::fwrite(&v, 2, 1, f); }
void put64(FILE* f, uint64_t v) { std::fwrite(&v, 8, 1, f); }

std::string sanitize(std::string s) {
    for (char& c : s)
        if (std::strchr("<>:\"/\\|?*", c) || static_cast<unsigned char>(c) < 32) c = '_';
    return s;
}

std::string joinPath(const std::string& dir, const std::string& file) {
    if (dir.empty()) return file;
    const char last = dir.back();
    return (last == '/' || last == '\\') ? dir + file : dir + "/" + file;
}

std::string timestamp() {
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H-%M-%S", &tm);
    return buf;
}
}  // namespace

bool WavWriter::open(const std::string& path, int channels, double rate) {
#ifdef _WIN32
    f_ = _wfopen(toWide(path).c_str(), L"wb");
#else
    f_ = std::fopen(path.c_str(), "wb");
#endif
    if (!f_) return false;
    path_ = path;
    channels_ = channels;
    rate_ = rate;
    dataBytes_ = 0;
    // RIFF header with a JUNK chunk reserved for an RF64 ds64 chunk.
    std::fwrite("RIFF", 1, 4, f_);
    put32(f_, 0);
    std::fwrite("WAVE", 1, 4, f_);
    std::fwrite("JUNK", 1, 4, f_);
    put32(f_, 28);
    for (int i = 0; i < 28; ++i) std::fputc(0, f_);
    std::fwrite("fmt ", 1, 4, f_);
    put32(f_, 16);
    put16(f_, 3);  // IEEE float
    put16(f_, static_cast<uint16_t>(channels));
    put32(f_, static_cast<uint32_t>(rate));
    put32(f_, static_cast<uint32_t>(rate) * channels * 4);
    put16(f_, static_cast<uint16_t>(channels * 4));
    put16(f_, 32);
    std::fwrite("data", 1, 4, f_);
    put32(f_, 0);
    return true;
}

void WavWriter::write(const float* interleaved, size_t frames) {
    if (!f_ || frames == 0) return;
    const size_t bytes = frames * static_cast<size_t>(channels_) * sizeof(float);
    std::fwrite(interleaved, 1, bytes, f_);
    dataBytes_ += bytes;
}

void WavWriter::close() {
    if (!f_) return;
    const uint64_t riffSize = dataBytes_ + 4 + (8 + 28) + (8 + 16) + 8;
    if (riffSize > 0xFFFFFFFFull) {
        // RF64: RIFF -> RF64, JUNK -> ds64 carrying the 64-bit sizes.
        std::fseek(f_, 0, SEEK_SET);
        std::fwrite("RF64", 1, 4, f_);
        put32(f_, 0xFFFFFFFF);
        std::fseek(f_, 12, SEEK_SET);
        std::fwrite("ds64", 1, 4, f_);
        put32(f_, 28);
        put64(f_, riffSize);
        put64(f_, dataBytes_);
        put64(f_, dataBytes_ / (static_cast<uint64_t>(channels_) * 4));
        put32(f_, 0);
        std::fseek(f_, 12 + 8 + 28 + 8 + 16 + 4, SEEK_SET);
        put32(f_, 0xFFFFFFFF);
    } else {
        std::fseek(f_, 4, SEEK_SET);
        put32(f_, static_cast<uint32_t>(riffSize));
        std::fseek(f_, 12 + 8 + 28 + 8 + 16 + 4, SEEK_SET);
        put32(f_, static_cast<uint32_t>(dataBytes_));
    }
    std::fclose(f_);
    f_ = nullptr;
}

// ------------------------------------------------------------------ recorder
Recorder::Recorder() { writer_ = std::thread([this] { writerLoop(); }); }

Recorder::~Recorder() {
    stop();
    running_ = false;
    if (writer_.joinable()) writer_.join();
    std::lock_guard lk(mu_);
    detachAllLocked();
}

void Recorder::detachAllLocked() {
    for (auto& [id, t] : tracks_)
        if (t.tap.rt) t.tap.rt->tap.store(nullptr, std::memory_order_release);
}

void Recorder::setTaps(const std::vector<Tap>& taps, double rate) {
    std::vector<Track> retired;
    {
        std::lock_guard lk(mu_);
        const bool rateChanged = rate != rate_;
        rate_ = rate;
        std::map<std::string, Track> next;
        for (const Tap& t : taps) {
            auto it = tracks_.find(t.nodeId);
            if (it != tracks_.end() && !rateChanged && it->second.tap.rt == t.rt && it->second.tap.channels == t.channels) {
                next[t.nodeId] = std::move(it->second);
                tracks_.erase(it);
                continue;
            }
            Track tr;
            tr.tap = t;
            tr.ring = std::make_unique<FrameRing>();
            tr.ring->init(static_cast<size_t>(t.channels), static_cast<size_t>(rate * 2));  // 2 s of slack for the writer
            const size_t frames = static_cast<size_t>(replaySeconds_.load() * rate);
            tr.replay.assign(frames * static_cast<size_t>(t.channels), 0.0f);
            next[t.nodeId] = std::move(tr);
        }
        // Detach rings that are no longer tapped; free them after the audio thread moved on.
        for (auto& [id, t] : tracks_) {
            if (t.tap.rt) t.tap.rt->tap.store(nullptr, std::memory_order_release);
            if (t.file) t.file->close();
            retired.push_back(std::move(t));
        }
        tracks_ = std::move(next);
        for (auto& [id, t] : tracks_) t.tap.rt->tap.store(t.ring.get(), std::memory_order_release);
    }
    if (!retired.empty()) std::this_thread::sleep_for(std::chrono::milliseconds(60));  // > one audio block
}

void Recorder::setReplaySeconds(int seconds) {
    std::lock_guard lk(mu_);
    const int s = std::clamp(seconds, 0, 3600);
    if (s == replaySeconds_) return;
    replaySeconds_ = s;
    for (auto& [id, t] : tracks_) {
        t.replay.assign(static_cast<size_t>(s * rate_) * static_cast<size_t>(t.tap.channels), 0.0f);
        t.replayPos = t.replayFill = 0;
    }
}

bool Recorder::start(const std::vector<std::string>& nodeIds, const std::string& folder, std::string& error) {
    std::lock_guard lk(mu_);
    if (recording_) return true;
    if (!makeDirs(folder)) {
        error = "Cannot create the recordings folder: " + folder;
        return false;
    }
    const std::string ts = timestamp();
    int opened = 0;
    for (const std::string& id : nodeIds) {
        auto it = tracks_.find(id);
        if (it == tracks_.end()) continue;
        Track& t = it->second;
        auto w = std::make_unique<WavWriter>();
        const std::string path = joinPath(folder, "NexTone " + ts + " - " + sanitize(t.tap.name) + ".wav");
        if (!w->open(path, t.tap.channels, rate_)) {
            error = "Cannot write " + path;
            continue;
        }
        t.ring->skip(t.ring->available());  // start "now"
        t.file = std::move(w);
        ++opened;
    }
    if (!opened) {
        if (error.empty()) error = "Nothing to record";
        return false;
    }
    startedAt_ = nowSeconds();
    recording_ = true;
    NT_INFO("Recording %d track(s) to %s", opened, folder.c_str());
    return true;
}

std::vector<std::string> Recorder::stop() {
    std::vector<std::string> files;
    std::lock_guard lk(mu_);
    if (!recording_) return files;
    recording_ = false;
    for (auto& [id, t] : tracks_) {
        if (!t.file) continue;
        // Flush what is still in the ring.
        std::vector<float> buf(4096 * static_cast<size_t>(t.tap.channels));
        size_t n;
        while ((n = t.ring->read(buf.data(), 4096)) > 0) t.file->write(buf.data(), n);
        t.file->close();
        files.push_back(t.file->path());
        t.file.reset();
    }
    NT_INFO("Recording stopped (%zu files)", files.size());
    return files;
}

std::vector<std::string> Recorder::saveReplay(int seconds, const std::string& folder, std::string& error) {
    std::vector<std::string> files;
    std::lock_guard lk(mu_);
    if (!makeDirs(folder)) {
        error = "Cannot create the recordings folder";
        return files;
    }
    const std::string ts = timestamp();
    for (auto& [id, t] : tracks_) {
        if (t.replay.empty() || t.replayFill == 0) continue;
        const size_t ch = static_cast<size_t>(t.tap.channels);
        const size_t cap = t.replay.size() / ch;
        const size_t want = std::min(t.replayFill, static_cast<size_t>(seconds * rate_));
        WavWriter w;
        const std::string path = joinPath(folder, "NexTone Replay " + ts + " - " + sanitize(t.tap.name) + ".wav");
        if (!w.open(path, t.tap.channels, rate_)) continue;
        size_t start = (t.replayPos + cap - want) % cap;
        size_t left = want;
        while (left > 0) {
            const size_t run = std::min(left, cap - start);
            w.write(&t.replay[start * ch], run);
            start = (start + run) % cap;
            left -= run;
        }
        w.close();
        files.push_back(path);
    }
    if (files.empty()) error = "The replay buffer is empty";
    return files;
}

void Recorder::writerLoop() {
    setThreadName("nt-recorder");
    std::vector<float> buf;
    while (running_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        std::lock_guard lk(mu_);
        for (auto& [id, t] : tracks_) {
            const size_t ch = static_cast<size_t>(t.tap.channels);
            buf.resize(4096 * ch);
            size_t n;
            while ((n = t.ring->read(buf.data(), 4096)) > 0) {
                if (t.file) t.file->write(buf.data(), n);
                if (!t.replay.empty()) {
                    const size_t cap = t.replay.size() / ch;
                    for (size_t i = 0; i < n; ++i) {
                        std::memcpy(&t.replay[t.replayPos * ch], &buf[i * ch], ch * sizeof(float));
                        t.replayPos = (t.replayPos + 1) % cap;
                    }
                    t.replayFill = std::min(cap, t.replayFill + n);
                }
            }
        }
    }
}

}  // namespace nt
