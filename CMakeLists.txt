cmake_minimum_required(VERSION 3.10)
project(VoxEngine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(vox_demo src/main.cpp)
target_include_directories(vox_demo PRIVATE src)
target_compile_options(vox_demo PRIVATE -O2 -fno-math-errno)

add_executable(vox_tui src/tui_main.cpp)
target_include_directories(vox_tui PRIVATE src)
target_compile_options(vox_tui PRIVATE -O2 -fno-math-errno)
