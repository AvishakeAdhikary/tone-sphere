# Developer / test tools
add_executable(audio-probe audio_probe.cpp)
target_compile_definitions(audio-probe PRIVATE UNICODE _UNICODE)
