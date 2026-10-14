# ToneSphere ASIO driver (COM in-proc server). Built for x64 and x86 (32-bit DAWs).
# Output name: ToneSphereASIO64.dll / ToneSphereASIO32.dll
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(TS_ASIO_NAME ToneSphereASIO64)
else()
  set(TS_ASIO_NAME ToneSphereASIO32)
endif()
add_library(tonesphere_asio MODULE tonesphere_asio.cpp tonesphere_asio.def)
set_target_properties(tonesphere_asio PROPERTIES OUTPUT_NAME ${TS_ASIO_NAME} PREFIX "")
target_include_directories(tonesphere_asio PRIVATE ${CMAKE_SOURCE_DIR}/third_party/asiosdk/common ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(tonesphere_asio PRIVATE UNICODE _UNICODE)
target_link_libraries(tonesphere_asio PRIVATE ole32 advapi32 avrt winmm user32 shell32)
