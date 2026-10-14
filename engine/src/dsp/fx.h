#pragma once
// Effect interface + built-in effects. Parameters are atomics so the control thread
// can change them while the audio thread runs; structural changes (add/remove)
// happen by publishing a new graph program.
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ts {

struct ParamDef {
    std::string id;
    std::string name;
    float min = 0, max = 1, def = 0;
    std::string unit;
    bool log = false;
    std::vector<std::string> choices;
};

class Fx {
public:
    explicit Fx(std::string type, std::string name, std::vector<ParamDef> defs);
    virtual ~Fx() = default;

    const std::string& type() const { return type_; }
    const std::string& name() const { return name_; }
    const std::vector<ParamDef>& paramDefs() const { return defs_; }
    std::string id;

    float param(size_t i) const { return values_[i].load(std::memory_order_relaxed); }
    bool setParam(const std::string& pid, float v);
    float paramById(const std::string& pid) const;

    std::atomic<bool> bypass{false};
    std::atomic<float> reductionDb{0.0f};
    /// Sidechain input (ducker): planar buffers of another node, set by the engine on publish.
    std::atomic<float* const*> sidechain{nullptr};
    std::atomic<int> sidechainChannels{0};
    std::string sidechainId;  // persisted node id

    /// Called on the control thread before the effect is published.
    virtual void prepare(double sampleRate, int maxBlock, int channels);
    /// Audio thread. In-place on planar buffers.
    virtual void process(float* const* buf, int channels, int frames) = 0;
    virtual bool hasEditor() const { return false; }
    virtual void openEditor() {}
    /// Plugin state for persistence (VST3), base64 or empty.
    virtual std::string saveState() const { return {}; }
    virtual void loadState(const std::string&) {}
    /// A user-facing problem with this effect (e.g. plugin missing or failed), or empty.
    virtual std::string status() const { return {}; }
    /// Instruments (VST3 plugins with a MIDI input) play notes sent here. Any thread except the
    /// audio thread; it never waits for audio. Status, data1, data2 as in a MIDI short message.
    virtual bool isInstrument() const { return false; }
    virtual void sendMidi(uint8_t status, uint8_t data1, uint8_t data2) {}
    std::string pluginId;

protected:
    double sr_ = 48000.0;
    int maxBlock_ = 1024;
    int channels_ = 2;

private:
    std::string type_;
    std::string name_;
    std::vector<ParamDef> defs_;
    std::unique_ptr<std::atomic<float>[]> values_;
};

/// Creates a built-in effect by type ("gate", "denoise", "eq", "compressor", "limiter", "ducker", "delay").
std::shared_ptr<Fx> createBuiltinFx(const std::string& type);

}  // namespace ts
