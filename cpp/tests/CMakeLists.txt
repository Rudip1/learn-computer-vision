find_package(Catch2 3 QUIET)
if(NOT Catch2_FOUND)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.7.1
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(Catch2)
  list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
endif()
include(Catch)

add_executable(mvg_tests
  test_projective.cpp
  test_camera.cpp
  test_calibration.cpp
)
target_link_libraries(mvg_tests PRIVATE mvg Catch2::Catch2WithMain)
catch_discover_tests(mvg_tests)
