cmake_minimum_required(VERSION 4.0)
project(CS219_ImageProcessLibrary)

set(CMAKE_C_STANDARD 23)
set(CMAKE_CXX_STANDARD 26)

add_executable(demo demo.cpp)

target_compile_options(demo PRIVATE -stdlib=libc++)
target_link_libraries(demo PRIVATE -lc++)