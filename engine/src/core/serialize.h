#pragma once
// JSON (de)serialisation of engine model pieces, shared by config.cpp and presets.cpp.
#include <string>

#include "core/engine.h"

namespace ts {

Bin binFromString(const std::string& s);
DeviceKind kindFromString(const std::string& s);
json settingsJson(const NodeSettingsData& s);
void settingsFrom(NodeSettingsData& s, const json& j);
json fxJson(const Fx& fx);
FxConfig fxConfigFrom(const json& j);
json routeJson(const RouteModel& r);
RouteModel routeFrom(const json& j);

}  // namespace ts
