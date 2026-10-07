foreach(example
    01_hierarchy
    02_camera
    03_calibrate
  )
  add_executable(${example} ${example}.cpp)
  target_link_libraries(${example} PRIVATE mvg)
  if(MVG_BUILD_TESTS)
    add_test(NAME example_${example} COMMAND ${example})
  endif()
endforeach()
