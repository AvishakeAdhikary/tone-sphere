# Developer / test tools
add_executable(audio-probe audio_probe.cpp)
target_compile_definitions(audio-probe PRIVATE UNICODE _UNICODE)

add_executable(asio-test-host asio_test_host.cpp)
target_include_directories(asio-test-host PRIVATE ${CMAKE_SOURCE_DIR}/third_party/asiosdk/common)
target_link_libraries(asio-test-host PRIVATE ole32 advapi32 user32)
