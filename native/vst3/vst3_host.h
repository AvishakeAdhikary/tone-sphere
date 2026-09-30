// The engine's side of VST3 hosting: an insert processor that runs an opened plugin on the
// audio thread. Everything else about plugins is behind the ts_vst3_* C functions.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace ts {

class Processor;

// A processor for the plugin opened as `handle`, sized for `channels` and `max_block`.
// Null, with `error` set, if the handle is unknown, crashed, or opened for another width.
std::unique_ptr<Processor> make_vst3_processor(uint32_t handle, uint32_t channels, uint32_t max_block,
                                               std::string& error);

}  // namespace ts
