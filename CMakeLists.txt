cmake_minimum_required(VERSION 3.25)
project(NexTone VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# We don't use C++20 modules; scanning also trips a Ninja lexing bug with some CMake versions.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(NEXTONE_BUILD_TESTS "Build unit tests" ON)
option(NEXTONE_BUILD_ASIO_DRIVER "Build the NexTone ASIO driver DLL (Windows)" ON)
option(NEXTONE_WITH_VST3 "Enable VST3 plugin hosting" ON)

if(MSVC)
  # Static CRT: the engine and ASIO driver must not depend on a VC++ redistributable.
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  add_compile_options(/W4 /permissive- /Zc:__cplusplus /utf-8 /MP /wd4100 /wd4324)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
else()
  add_compile_options(-Wall -Wextra -Wno-unused-parameter)
endif()

include(FetchContent)
set(FETCHCONTENT_QUIET ON)
FetchContent_Declare(nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(nlohmann_json)

if(NEXTONE_BUILD_TESTS)
  FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.9.1
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(Catch2)
  enable_testing()
endif()

add_subdirectory(engine)
if(WIN32)
  if(NEXTONE_BUILD_ASIO_DRIVER)
    add_subdirectory(drivers/win/asio)
  endif()
  add_subdirectory(tools)
endif()
