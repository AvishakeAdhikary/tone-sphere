# NexTone audio engine

# ---- RNNoise (BSD-3) for the noise-suppression effect; v0.1.1 ships its model in-tree.
FetchContent_Declare(rnnoise
  GIT_REPOSITORY https://github.com/xiph/rnnoise.git
  GIT_TAG v0.1.1
  GIT_SHALLOW TRUE)
FetchContent_GetProperties(rnnoise)
if(NOT rnnoise_POPULATED)
  FetchContent_Populate(rnnoise)
endif()
set(RNNOISE_PATCHED ${CMAKE_CURRENT_BINARY_DIR}/rnnoise_patched)
if(MSVC)
  execute_process(COMMAND ${CMAKE_COMMAND} -DSRC_DIR=${rnnoise_SOURCE_DIR} -DOUT_DIR=${RNNOISE_PATCHED}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/patch_rnnoise.cmake)
  set(RNNOISE_PITCH ${RNNOISE_PATCHED}/pitch.c)
  set(RNNOISE_LPC ${RNNOISE_PATCHED}/celt_lpc.c)
else()
  set(RNNOISE_PITCH ${rnnoise_SOURCE_DIR}/src/pitch.c)
  set(RNNOISE_LPC ${rnnoise_SOURCE_DIR}/src/celt_lpc.c)
endif()
add_library(rnnoise STATIC
  ${rnnoise_SOURCE_DIR}/src/denoise.c
  ${rnnoise_SOURCE_DIR}/src/rnn.c
  ${rnnoise_SOURCE_DIR}/src/rnn_data.c
  ${RNNOISE_PITCH}
  ${rnnoise_SOURCE_DIR}/src/kiss_fft.c
  ${RNNOISE_LPC})
target_include_directories(rnnoise PUBLIC ${rnnoise_SOURCE_DIR}/include PRIVATE ${rnnoise_SOURCE_DIR}/src)
if(MSVC)
  target_compile_options(rnnoise PRIVATE /W0)
  target_compile_definitions(rnnoise PRIVATE _USE_MATH_DEFINES)
else()
  target_compile_options(rnnoise PRIVATE -w)
endif()

# ---- real-time core (platform independent, unit tested)
add_library(nextone_core STATIC
  src/core/resampler.cpp
  src/core/bridge.cpp
)
target_include_directories(nextone_core PUBLIC src)

# ---- engine library
set(ENGINE_SOURCES
  src/core/graph.cpp
  src/core/engine.cpp
  src/core/config.cpp
  src/core/state.cpp
  src/core/rpc.cpp
  src/dsp/fx.cpp
  src/backends/device.cpp
  src/ipc/ws_server.cpp
  src/util/log.cpp
  src/util/platform.cpp
  src/util/encoding.cpp
)
if(WIN32)
  list(APPEND ENGINE_SOURCES
    src/backends/win/wasapi.cpp
    src/backends/win/app_audio.cpp
    src/backends/win/vad.cpp
    src/backends/win/asio_clients.cpp
    src/backends/win/asio_host.cpp)
endif()
add_library(nextone_engine_lib STATIC ${ENGINE_SOURCES})
target_link_libraries(nextone_engine_lib PUBLIC nextone_core nlohmann_json::nlohmann_json rnnoise)
target_compile_definitions(nextone_engine_lib PUBLIC NEXTONE_WITH_RNNOISE)
if(WIN32)
  target_compile_definitions(nextone_engine_lib PUBLIC UNICODE _UNICODE)
  target_include_directories(nextone_engine_lib PRIVATE ${CMAKE_SOURCE_DIR}/third_party/asiosdk/common)
  target_link_libraries(nextone_engine_lib PUBLIC winmm avrt ole32 ws2_32 mmdevapi runtimeobject version windowscodecs shell32)
else()
  find_package(Threads REQUIRED)
  target_link_libraries(nextone_engine_lib PUBLIC Threads::Threads)
endif()

add_executable(nextone-engine src/main.cpp)
target_link_libraries(nextone-engine PRIVATE nextone_engine_lib)
if(WIN32)
  # GUI subsystem (no console window) but keep the standard main() entry point.
  set_target_properties(nextone-engine PROPERTIES WIN32_EXECUTABLE ON LINK_FLAGS "/ENTRY:mainCRTStartup")
  target_sources(nextone-engine PRIVATE src/engine.rc)
endif()

if(NEXTONE_BUILD_TESTS)
  add_executable(nextone_core_tests tests/core_tests.cpp)
  target_link_libraries(nextone_core_tests PRIVATE nextone_core Catch2::Catch2WithMain)
  add_test(NAME core COMMAND nextone_core_tests)

  add_executable(nextone_engine_tests tests/engine_tests.cpp)
  target_link_libraries(nextone_engine_tests PRIVATE nextone_engine_lib Catch2::Catch2WithMain)
  add_test(NAME engine COMMAND nextone_engine_tests)
endif()
