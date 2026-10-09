#pragma once
// Base64 and SHA-1 (for data URLs and the WebSocket handshake).
#include <array>
#include <cstdint>
#include <string>
#include <string_view>

namespace nt {

std::string base64Encode(const void* data, size_t len);
std::string base64Encode(std::string_view s);
std::array<uint8_t, 20> sha1(std::string_view s);

}  // namespace nt
