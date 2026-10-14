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

# ---- VST3 SDK (MIT since 3.8) for plugin hosting; only the parts we need are compiled.
FetchContent_Declare(vst3sdk
  GIT_REPOSITORY https://github.com/steinbergmedia/vst3sdk.git
  GIT_TAG v3.8.1_build_84
  GIT_SHALLOW TRUE
  GIT_SUBMODULES base pluginterfaces public.sdk)
FetchContent_GetProperties(vst3sdk)
if(NOT vst3sdk_POPULATED)
  FetchContent_Populate(vst3sdk)
endif()
set(VST3_SDK ${vst3sdk_SOURCE_DIR} CACHE INTERNAL "")
set(VST3_PUB ${VST3_SDK}/public.sdk/source)
# interfaces + base classes shared by the host and the test plugin
add_library(vst3_base STATIC
  ${VST3_SDK}/pluginterfaces/base/coreiids.cpp
  ${VST3_SDK}/pluginterfaces/base/funknown.cpp
  ${VST3_SDK}/pluginterfaces/base/ustring.cpp
  ${VST3_SDK}/pluginterfaces/base/conststringtable.cpp
  ${VST3_SDK}/base/source/baseiids.cpp
  ${VST3_SDK}/base/source/fobject.cpp
  ${VST3_SDK}/base/source/fstring.cpp
  ${VST3_SDK}/base/source/fdebug.cpp
  ${VST3_SDK}/base/source/fbuffer.cpp
  ${VST3_SDK}/base/source/fstreamer.cpp
  ${VST3_SDK}/base/source/updatehandler.cpp
  ${VST3_SDK}/base/thread/source/flock.cpp
  ${VST3_SDK}/base/thread/source/fcondition.cpp
  ${VST3_PUB}/common/commoniids.cpp
  ${VST3_PUB}/common/memorystream.cpp
  ${VST3_PUB}/common/commonstringconvert.cpp
  ${VST3_PUB}/vst/vstinitiids.cpp
  ${VST3_PUB}/vst/utility/stringconvert.cpp
  $<IF:$<BOOL:${WIN32}>,${VST3_PUB}/common/threadchecker_win32.cpp,${VST3_PUB}/common/threadchecker_linux.cpp>)
target_include_directories(vst3_base PUBLIC ${VST3_SDK})
target_compile_definitions(vst3_base PUBLIC $<IF:$<CONFIG:Debug>,DEVELOPMENT=1,RELEASE=1>)
target_compile_options(vst3_base PRIVATE $<IF:$<CXX_COMPILER_ID:MSVC>,/W0,-w>)
set(VST3_MODULE_SRC ${VST3_PUB}/vst/hosting/module_linux.cpp)
if(WIN32)
  set(VST3_MODULE_SRC ${VST3_PUB}/vst/hosting/module_win32.cpp)
endif()
add_library(vst3_hosting STATIC
  ${VST3_PUB}/vst/hosting/module.cpp
  ${VST3_MODULE_SRC}
  ${VST3_PUB}/vst/hosting/hostclasses.cpp
  ${VST3_PUB}/vst/hosting/pluginterfacesupport.cpp
  ${VST3_PUB}/vst/hosting/plugprovider.cpp
  ${VST3_PUB}/vst/hosting/processdata.cpp
  ${VST3_PUB}/vst/hosting/parameterchanges.cpp
  ${VST3_PUB}/vst/hosting/eventlist.cpp
  ${VST3_PUB}/vst/hosting/connectionproxy.cpp)
target_link_libraries(vst3_hosting PUBLIC vst3_base)
target_compile_options(vst3_hosting PRIVATE $<IF:$<CXX_COMPILER_ID:MSVC>,/W0,-w>)
if(NOT WIN32)
  target_link_libraries(vst3_hosting PUBLIC ${CMAKE_DL_LIBS})
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
  src/core/recorder.cpp
  src/core/latency.cpp
  src/core/presets.cpp
  src/control/remote.cpp
  src/dsp/fx.cpp
  src/backends/device.cpp
  src/ipc/ws_server.cpp
  src/util/log.cpp
  src/util/platform.cpp
  src/util/encoding.cpp
  src/util/ui_thread.cpp
  src/net/vban.cpp
  src/plugins/vst3_host.cpp
)
if(WIN32)
  list(APPEND ENGINE_SOURCES
    src/backends/win/wasapi.cpp
    src/backends/win/app_audio.cpp
    src/backends/win/vad.cpp
    src/backends/win/asio_clients.cpp
    src/backends/win/asio_host.cpp
    src/backends/win/peak_meters.cpp
    src/backends/win/hotkeys.cpp)
endif()
add_library(nextone_engine_lib STATIC ${ENGINE_SOURCES})
target_link_libraries(nextone_engine_lib PUBLIC nextone_core nlohmann_json::nlohmann_json rnnoise vst3_hosting)
target_compile_definitions(nextone_engine_lib PUBLIC NEXTONE_WITH_RNNOISE)
if(WIN32)
  target_compile_definitions(nextone_engine_lib PUBLIC UNICODE _UNICODE)
  target_include_directories(nextone_engine_lib PRIVATE ${CMAKE_SOURCE_DIR}/third_party/asiosdk/common)
  target_link_libraries(nextone_engine_lib PUBLIC winmm avrt ole32 ws2_32 mmdevapi runtimeobject version windowscodecs shell32)
else()
  find_package(Threads REQUIRED)
  target_link_libraries(nextone_engine_lib PUBLIC Threads::Threads)
  if(NOT APPLE)
    find_package(PkgConfig)
    if(PkgConfig_FOUND)
      pkg_check_modules(PIPEWIRE IMPORTED_TARGET libpipewire-0.3)
    endif()
    if(PIPEWIRE_FOUND)
      target_sources(nextone_engine_lib PRIVATE src/backends/linux/pipewire.cpp)
      target_link_libraries(nextone_engine_lib PUBLIC PkgConfig::PIPEWIRE)
      target_compile_definitions(nextone_engine_lib PUBLIC NEXTONE_WITH_PIPEWIRE)
    else()
      message(WARNING "libpipewire-0.3 not found: building without the PipeWire backend")
    endif()
  endif()
endif()

add_executable(nextone-engine src/main.cpp)
target_link_libraries(nextone-engine PRIVATE nextone_engine_lib)
if(WIN32)
  # GUI subsystem (no console window) but keep the standard main() entry point.
  set_target_properties(nextone-engine PROPERTIES WIN32_EXECUTABLE ON LINK_FLAGS "/ENTRY:mainCRTStartup")
  target_sources(nextone-engine PRIVATE src/engine.rc)
endif()

# Out-of-process plugin scanner (a crashing plugin cannot take the engine down while scanning).
add_executable(nextone-plugscan src/plugins/plugscan_main.cpp)
target_link_libraries(nextone-plugscan PRIVATE nextone_engine_lib)

# Small VST3 effect used by the automated tests (gain + a native editor window).
if(WIN32)
  add_library(nextone_test_gain MODULE
    tests/test_plugin/test_gain.cpp
    ${VST3_PUB}/main/dllmain.cpp
    ${VST3_PUB}/main/pluginfactory.cpp
    ${VST3_PUB}/main/moduleinit.cpp
    ${VST3_PUB}/vst/vstsinglecomponenteffect.cpp
    ${VST3_PUB}/vst/vstcomponentbase.cpp
    ${VST3_PUB}/vst/vstbus.cpp
    ${VST3_PUB}/vst/vstparameters.cpp
    ${VST3_PUB}/common/pluginview.cpp)
  target_link_libraries(nextone_test_gain PRIVATE vst3_base comctl32)
  set_target_properties(nextone_test_gain PROPERTIES PREFIX "" SUFFIX ".vst3"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/test-plugins)
endif()

if(NEXTONE_BUILD_TESTS)
  add_executable(nextone_core_tests tests/core_tests.cpp)
  target_link_libraries(nextone_core_tests PRIVATE nextone_core Catch2::Catch2WithMain)
  add_test(NAME core COMMAND nextone_core_tests)

  add_executable(nextone_engine_tests tests/engine_tests.cpp)
  target_link_libraries(nextone_engine_tests PRIVATE nextone_engine_lib Catch2::Catch2WithMain)
  add_test(NAME engine COMMAND nextone_engine_tests)
endif()
