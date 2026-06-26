cmake_minimum_required(VERSION 3.16)
project(pc_audio_dsp C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -Wall -Wextra -include stddef.h")

find_package(PkgConfig REQUIRED)
pkg_check_modules(SNDFILE REQUIRED sndfile)

include_directories(dsp ${SNDFILE_INCLUDE_DIRS})
file(GLOB DSP_SOURCES "dsp/*.c")

add_executable(dsp_app main.c ${DSP_SOURCES})

# FIXED: Added 'm' to link the standard Linux math library explicitly
target_link_libraries(dsp_app ${SNDFILE_LIBRARIES} m)