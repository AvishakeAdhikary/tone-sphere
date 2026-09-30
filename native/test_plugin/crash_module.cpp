// A "VST3 module" whose factory entry point faults: what a broken or incompatible plugin
// does to a host that loads it in-process. The scanner must survive it (it runs in a
// subprocess for exactly this), and in-process loading must at least report the fault.
#include <windows.h>

extern "C" __declspec(dllexport) bool InitDll() { return true; }
extern "C" __declspec(dllexport) bool ExitDll() { return true; }

extern "C" __declspec(dllexport) void* GetPluginFactory() {
    volatile int* nowhere = nullptr;
    *nowhere = 1;
    return nullptr;
}
