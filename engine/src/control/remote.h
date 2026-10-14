#pragma once
// Remote control: OSC over UDP and MIDI controllers (Windows).
//
// OSC (off unless EngineSettings::oscPort is set), node = id or name:
//   /tonesphere/volume ,sf  <node> <0..1>     fader position (0 = silent, 1 = +6 dB)
//   /tonesphere/gain   ,sf  <node> <dB>
//   /tonesphere/mute   ,s[i|f|T|F] <node> [0/1]   (no value = toggle)
//   /tonesphere/solo   ,s[i|f|T|F] <node> [0/1]
//   /tonesphere/scene  ,s   <scene id or name>
//   /tonesphere/record [,i] [0/1]               (no value = toggle)
//   /tonesphere/replay                          (save the instant replay)
// MIDI: EngineSettings::midiMappings, e.g.
//   {"type":"cc","channel":1,"number":7,"action":"volume","target":"bus_1"}
//   {"type":"note","channel":0,"number":60,"action":"mute","target":"Microphone"}   (channel 0 = any)
// Actions run on a worker thread through Engine::call (never under the engine lock).
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace ts {

class Engine;

struct RemoteAction {
    std::string action;  // volume | gain | mute | solo | scene | record | replay
    std::string target;
    std::optional<float> value;  // none = toggle (mute/solo/record)
};

struct OscArg {
    char type = 0;  // i f s T F d
    double num = 0;
    std::string str;
};

/// Parses one OSC message (bundles are unpacked into several). Exposed for tests.
bool parseOscPacket(const uint8_t* data, size_t len, std::vector<std::pair<std::string, std::vector<OscArg>>>& out);
/// Maps an OSC message to an action.
std::optional<RemoteAction> oscToAction(const std::string& address, const std::vector<OscArg>& args);
/// Maps a MIDI short message (status, data1, data2) to an action using the mappings.
std::optional<RemoteAction> midiToAction(const nlohmann::json& mappings, uint8_t status, uint8_t d1, uint8_t d2);
/// Fader position (0..1) to gain in dB.
float faderToDb(float pos);

class RemoteControl {
public:
    explicit RemoteControl(Engine& engine);
    ~RemoteControl();
    /// Waits for the next MIDI control message (MIDI learn) and returns {type, channel, number, value}.
    nlohmann::json learn(double seconds);
    /// Connected MIDI input devices.
    nlohmann::json midiDevices();

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace ts
