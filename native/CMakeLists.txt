cmake_minimum_required(VERSION 3.25)
project(tonesphere_native LANGUAGES C CXX)

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

add_library(tonesphere_native SHARED
    engine/abi.cpp
    engine/engine.cpp
    engine/rt_alloc.cpp
    $<$<BOOL:${WIN32}>:windows_audio/wasapi.cpp>
)
target_include_directories(tonesphere_native PRIVATE include engine)
if(WIN32)
    target_link_libraries(tonesphere_native PRIVATE ole32 avrt mmdevapi propsys)
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
