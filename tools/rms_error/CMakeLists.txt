cmake_minimum_required(VERSION 3.10)

project(rms_tool C)

set(CMAKE_C_STANDARD 99)

add_library(rms rms_error.c)

add_executable(main main.c)

target_link_libraries(main rms m)
