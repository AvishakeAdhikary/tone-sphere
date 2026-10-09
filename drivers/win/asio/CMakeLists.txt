# NexTone ASIO driver (COM in-proc server). Built for x64 and x86 (32-bit DAWs).
# Output name: NexToneASIO64.dll / NexToneASIO32.dll
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(NT_ASIO_NAME NexToneASIO64)
else()
  set(NT_ASIO_NAME NexToneASIO32)
endif()
add_library(nextone_asio MODULE nextone_asio.cpp nextone_asio.def)
set_target_properties(nextone_asio PROPERTIES OUTPUT_NAME ${NT_ASIO_NAME} PREFIX "")
target_include_directories(nextone_asio PRIVATE ${CMAKE_SOURCE_DIR}/third_party/asiosdk/common ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(nextone_asio PRIVATE UNICODE _UNICODE)
target_link_libraries(nextone_asio PRIVATE ole32 advapi32 avrt winmm user32 shell32)
