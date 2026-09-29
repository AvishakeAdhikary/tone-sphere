#pragma once

#include <cstdint>
#include <memory>

namespace ts {

class Engine;
class DeviceBackend;

// Runs the engine's blocks from a timer thread: the backend for a plan that routes no device.
std::unique_ptr<DeviceBackend> start_clock(Engine& engine, uint32_t block_frames);

}  // namespace ts
