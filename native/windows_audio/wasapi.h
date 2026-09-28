// WASAPI: endpoint enumeration, device-change notifications, and the device backend that
// drives the engine's audio thread.
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "tonesphere_native.h"

namespace ts {

class Engine;
class DeviceBackend;

namespace wasapi {

int32_t enumerate(ts_device_info* out, int32_t capacity, std::string& error);

// Registers an IMMNotificationClient and queues what it reports. Notifications arrive on
// system threads, never the audio thread, so a mutex-guarded queue is fine here.
class DeviceWatcher {
public:
    ~DeviceWatcher();
    bool start(std::string& error);
    void stop();
    int32_t poll(ts_device_event* out, int32_t capacity);
    void push(const ts_device_event& event);

private:
    std::mutex mutex_;
    std::vector<ts_device_event> queue_;
    void* enumerator_ = nullptr;
    void* client_ = nullptr;
    uint32_t dropped_ = 0;
};

// Starts the streams and runs the engine's blocks from the master stream's thread.
std::unique_ptr<DeviceBackend> start(Engine& engine, const ts_stream_desc* streams, uint32_t count,
                                     uint32_t master, std::string& error);

}  // namespace wasapi
}  // namespace ts
