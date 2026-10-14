#pragma once
// VBAN (VB-Audio network audio protocol) — audio sub-protocol, PCM codec.
// Compatible with VoiceMeeter, VBAN Receptor/Talkie and other ToneSphere computers.
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "backends/device.h"

namespace ts::vban {

constexpr size_t kHeaderBytes = 28;
constexpr size_t kMaxPayload = 1436;
constexpr int kMaxSamplesPerPacket = 256;
constexpr int kDefaultPort = 6980;

enum class DataType : uint8_t { Int8 = 0, Int16 = 1, Int24 = 2, Int32 = 3, Float32 = 4, Float64 = 5 };

struct Header {
    double sampleRate = 48000;
    int samples = 0;   // frames in this packet (1..256)
    int channels = 2;  // 1..256
    DataType type = DataType::Int16;
    std::string stream;  // up to 16 chars
    uint32_t frame = 0;
};

/// Returns the VBAN sample-rate index for a rate, or -1 if VBAN can't carry it.
int rateIndex(double rate);
double rateFromIndex(int idx);
int bytesPerSample(DataType t);
/// Largest frame count per packet for the given channels/type.
int framesPerPacket(int channels, DataType t);

/// Encodes one audio packet from interleaved float frames. Returns the packet size.
size_t encode(const Header& h, const float* interleaved, uint8_t* out, size_t outCap);
/// Parses a packet; returns false for non-audio or malformed packets.
bool parse(const uint8_t* pkt, size_t len, Header& h, const uint8_t*& payload, size_t& payloadBytes);
/// Decodes `frames` frames of payload into interleaved floats with `outChannels` channels.
void decode(const Header& h, const uint8_t* payload, float* out, int outChannels);

/// Network backend: VBAN streams appear as virtual devices.
class VbanBackend final : public Backend {
public:
    VbanBackend();
    ~VbanBackend() override;
    const char* name() const override { return "vban"; }
    std::vector<DeviceInfo> enumerate() override { return {}; }  // streams are user-created nodes
    std::unique_ptr<DeviceStream> open(const DeviceInfo& dev, const StreamRequest& req, std::string& error) override;
};

}  // namespace ts::vban
