#pragma once
// Portable routing presets ("maps"): devices are referred to by bin, kind and name patterns instead
// of machine-specific ids, so a preset works on any PC and with devices that are plugged in later.
//
//   {"format": "tonesphere-preset", "version": 1, "name": "...", "description": "...",
//    "devices": [{"key": "guitar", "bin": "physical-in", "kind": "wasapi",
//                 "match": ["Line (AI-04)", "AI-04"], "label": "...", "channels": 2,
//                 "gainDb": 0, "muted": false, "settings": {...}, "fx": [...]},
//                {"key": "rig-in", "bin": "app-in", "kind": "asioclient", "exe": "guitar rig 7.exe"}],
//    "mixes":   [{"key": "stream", "name": "Stream mix", "channels": 2, "gainDb": 0, "fx": [...]}],
//    "routes":  [{"from": "guitar", "to": "rig-in", "channels": [[0, 0], [0, 1]], "gainDb": 0}]}
#include <string>

#include <nlohmann/json.hpp>

namespace ts {

/// "Name (Subtitle)": how Windows names endpoints, e.g. "Line (AI-04)".
std::string deviceLabel(const std::string& name, const std::string& subtitle);

/// How well a device matches a preset device rule (0 = not at all). Earlier patterns and exact
/// matches score higher: exact label 3, exact name 2, label contains the pattern 1.
int presetMatchScore(const nlohmann::json& rule, const std::string& bin, const std::string& kind, const std::string& name,
                     const std::string& subtitle);

}  // namespace ts
