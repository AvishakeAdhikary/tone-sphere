#pragma once
// Base64 and SHA-1 (data URLs, plugin state and the WebSocket handshake).
#include <array>
#include <cstdint>
#include <string>
#include <string_view>

namespace nt {

std::string base64Encode(const void* data, size_t len);
std::string base64Encode(std::string_view s);
/// Decodes standard base64 (padding optional, whitespace ignored); invalid characters are skipped.
std::string base64Decode(std::string_view s);
std::array<uint8_t, 20> sha1(std::string_view s);

}  // namespace nt
