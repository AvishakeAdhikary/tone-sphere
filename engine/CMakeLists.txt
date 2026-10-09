# NexTone audio engine
add_library(nextone_core STATIC
  src/core/resampler.cpp
  src/core/bridge.cpp
)
target_include_directories(nextone_core PUBLIC src)

if(NEXTONE_BUILD_TESTS)
  add_executable(nextone_core_tests tests/core_tests.cpp)
  target_link_libraries(nextone_core_tests PRIVATE nextone_core Catch2::Catch2WithMain)
  add_test(NAME core COMMAND nextone_core_tests)
endif()
