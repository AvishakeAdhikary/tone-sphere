cmake_minimum_required(VERSION 3.25)
project(tonesphere_native VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# Static CRT: the DLLs must load on a machine with no Visual C++ redistributable, and the
# replaced operator new in engine/rt_alloc.cpp must be this DLL's own, not a shared CRT's.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

set(TS_OUTPUT_DIR "${CMAKE_BINARY_DIR}/bin" CACHE PATH "Where the built DLLs go")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${TS_OUTPUT_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${TS_OUTPUT_DIR}")

if(MSVC)
    # C4324 reports padding added by alignas — which is the point of the cache-line
    # alignment in engine/spsc.h, not a mistake.
    add_compile_options(/W4 /permissive- /utf-8 /EHsc /Zc:__cplusplus /MP /wd4324)
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
endif()

# The Steinberg VST3 SDK (MIT), fetched into sdks/vst3sdk by scripts/fetch_sdks.py. Only
# what hosting needs: no examples, no VSTGUI, and never the plug-in symlink into the
# system VST3 folder, which needs administrator rights.
set(TS_VST3_SDK "${CMAKE_CURRENT_SOURCE_DIR}/../sdks/vst3sdk" CACHE PATH "Steinberg VST3 SDK root")
set(TS_WITH_VST3 OFF)
if(WIN32 AND EXISTS "${TS_VST3_SDK}/CMakeLists.txt")
    set(TS_WITH_VST3 ON)
    set(SMTG_ENABLE_VST3_PLUGIN_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(SMTG_ENABLE_VST3_HOSTING_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(SMTG_ENABLE_VSTGUI_SUPPORT OFF CACHE BOOL "" FORCE)
    set(SMTG_CREATE_PLUGIN_LINK OFF CACHE BOOL "" FORCE)
    set(SMTG_RUN_VST_VALIDATOR OFF CACHE BOOL "" FORCE)
    set(SMTG_USE_STATIC_CRT ON CACHE BOOL "" FORCE)
    # The SDK places a plug-in's binary inside its bundle only for configurations listed
    # here, which a single-config generator such as Ninja leaves empty.
    set(CMAKE_CONFIGURATION_TYPES "Debug;Release")
    add_subdirectory("${TS_VST3_SDK}" vst3sdk EXCLUDE_FROM_ALL)
    message(STATUS "VST3 host: building against ${TS_VST3_SDK}")
else()
    message(STATUS "VST3 host: not built (no VST3 SDK at ${TS_VST3_SDK}; run scripts/fetch_sdks.py)")
endif()

add_library(tonesphere_native SHARED
    engine/abi.cpp
    engine/engine.cpp
    engine/rt_alloc.cpp
    $<$<BOOL:${WIN32}>:windows_audio/wasapi.cpp>
    $<$<BOOL:${WIN32}>:engine/clock.cpp>
)
target_include_directories(tonesphere_native PRIVATE include engine)
if(WIN32)
    target_link_libraries(tonesphere_native PRIVATE ole32 avrt mmdevapi propsys)
endif()
if(TS_WITH_VST3)
    target_sources(tonesphere_native PRIVATE
        vst3/vst3_host.cpp
        "${TS_VST3_SDK}/public.sdk/source/vst/hosting/module_win32.cpp"
        "${TS_VST3_SDK}/public.sdk/source/vst/hosting/plugprovider.cpp"
        "${TS_VST3_SDK}/public.sdk/source/common/memorystream.cpp"
    )
    target_include_directories(tonesphere_native PRIVATE vst3 "${TS_VST3_SDK}")
    target_compile_definitions(tonesphere_native PRIVATE TS_WITH_VST3)
    target_link_libraries(tonesphere_native PRIVATE sdk_hosting)
    # The SDK's own sources are not held to this project's /WX.
    set_source_files_properties(
        "${TS_VST3_SDK}/public.sdk/source/vst/hosting/module_win32.cpp"
        "${TS_VST3_SDK}/public.sdk/source/vst/hosting/plugprovider.cpp"
        "${TS_VST3_SDK}/public.sdk/source/common/memorystream.cpp"
        PROPERTIES COMPILE_OPTIONS "/WX-;/W3")

    # The deterministic test plugin, and a module that crashes on load, for tests/native.
    set(TS_TEST_PLUGINS "${CMAKE_BINARY_DIR}/test_plugins")
    smtg_add_vst3plugin(tonesphere_test_gain test_plugin/test_plugin.cpp)
    target_link_libraries(tonesphere_test_gain PRIVATE sdk)
    add_library(tonesphere_test_crash_module SHARED test_plugin/crash_module.cpp)
    set_target_properties(tonesphere_test_crash_module PROPERTIES
        OUTPUT_NAME "ToneSphereTestCrashModule" SUFFIX ".vst3"
        RUNTIME_OUTPUT_DIRECTORY "${TS_TEST_PLUGINS}/ToneSphereTestCrashModule.vst3/Contents/x86_64-win")
endif()
target_compile_definitions(tonesphere_native PRIVATE TONESPHERE_NATIVE_BUILD)
if(MSVC)
    target_compile_options(tonesphere_native PRIVATE /WX)
endif()

# The ASIO host: GPLv3, its own DLL, built only when the ASIO SDK has been fetched into
# sdks/asiosdk (scripts/fetch_sdks.py). Everything else builds and runs without it.
set(TS_ASIO_SDK "${CMAKE_CURRENT_SOURCE_DIR}/../sdks/asiosdk" CACHE PATH "Steinberg ASIO SDK root")
if(WIN32 AND EXISTS "${TS_ASIO_SDK}/common/iasiodrv.h")
    add_library(tonesphere_asio SHARED asio/asio_host.cpp)
    target_include_directories(tonesphere_asio PRIVATE include engine asio "${TS_ASIO_SDK}/common")
    target_compile_definitions(tonesphere_asio PRIVATE TONESPHERE_ASIO_BUILD)
    target_link_libraries(tonesphere_asio PRIVATE tonesphere_native ole32 user32 advapi32 avrt)
    if(MSVC)
        target_compile_options(tonesphere_asio PRIVATE /WX)
    endif()
    message(STATUS "ASIO host: building against ${TS_ASIO_SDK}")
else()
    message(STATUS "ASIO host: not built (no ASIO SDK at ${TS_ASIO_SDK}; run scripts/fetch_sdks.py)")
endif()
