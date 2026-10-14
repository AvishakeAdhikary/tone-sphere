cmake_minimum_required(VERSION 3.25)
# VERSION is the single source of the version number (CI stamps the released one).
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/VERSION TS_VERSION LIMIT_COUNT 1)
project(ToneSphere VERSION ${TS_VERSION} LANGUAGES C CXX)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/engine/src/version.h.in ${CMAKE_BINARY_DIR}/generated/version.h @ONLY)
include_directories(${CMAKE_BINARY_DIR}/generated)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# We don't use C++20 modules; scanning also trips a Ninja lexing bug with some CMake versions.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(TONESPHERE_BUILD_TESTS "Build unit tests" ON)
option(TONESPHERE_BUILD_ASIO_DRIVER "Build the ToneSphere ASIO driver DLL (Windows)" ON)
option(TONESPHERE_WITH_VST3 "Enable VST3 plugin hosting" ON)

if(MSVC)
  # Static CRT: the engine and ASIO driver must not depend on a VC++ redistributable.
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  add_compile_options(/W4 /permissive- /Zc:__cplusplus /utf-8 /MP /wd4100 /wd4324)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
else()
  add_compile_options(-Wall -Wextra -Wno-unused-parameter)
endif()

# Release builds optimise across files (MSVC /GL + /LTCG, -flto): the audio path inlines through
# the DSP, graph and bridge code.
include(CheckIPOSupported)
check_ipo_supported(RESULT TS_IPO LANGUAGES C CXX)
option(TONESPHERE_LTO "Link-time optimisation for release builds" ON)
if(TONESPHERE_LTO AND TS_IPO)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

include(FetchContent)
set(FETCHCONTENT_QUIET ON)
FetchContent_Declare(nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(nlohmann_json)

if(TONESPHERE_BUILD_TESTS)
  FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.9.1
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(Catch2)
  enable_testing()
endif()

add_subdirectory(engine)
if(WIN32)
  if(TONESPHERE_BUILD_ASIO_DRIVER)
    add_subdirectory(drivers/win/asio)
  endif()
  add_subdirectory(tools)
endif()
