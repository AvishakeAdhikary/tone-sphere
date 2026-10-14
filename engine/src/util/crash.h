#pragma once
// Crash reports: if the engine dies on an unexpected fault, it leaves a report in
// <data dir>/crashes so the next start can tell the user, and a developer can see where it failed.
//   Windows: crash-<time>.txt (fault, address, module, version) + crash-<time>.dmp (minidump).
//   Linux:   crash-<time>.txt (signal, version).
#include <string>

namespace ts {

/// Installs the process-wide handler. `dir` is created now (nothing is allocated while crashing).
void installCrashHandler(const std::string& dir, const char* version);

}  // namespace ts
