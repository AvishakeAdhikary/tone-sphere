#include "backends/device.h"

namespace nt {

const char* deviceKindName(DeviceKind k) {
    switch (k) {
        case DeviceKind::Wasapi: return "wasapi";
        case DeviceKind::Asio: return "asio";
        case DeviceKind::Vad: return "vad";
        case DeviceKind::Compat: return "compat";
        case DeviceKind::App: return "app";
        case DeviceKind::AsioClient: return "asioclient";
        case DeviceKind::Vban: return "vban";
        case DeviceKind::PipeWire: return "pipewire";
        case DeviceKind::CoreAudio: return "coreaudio";
    }
    return "?";
}

}  // namespace nt
