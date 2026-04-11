cmake_minimum_required(VERSION 3.10)

project(DSP_Audio_Project C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Force console application
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /SUBSYSTEM:CONSOLE")

# Include directory
include_directories(${CMAKE_SOURCE_DIR}/inc)

# Collect all .c files
file(GLOB SOURCES "${CMAKE_SOURCE_DIR}/src/*.c")

add_executable(dsp_test ${SOURCES})

# === IMPORTANT: Do NOT link 'm' on Windows ===
# target_link_libraries(dsp_test m)   ← Remove or comment this line

# Warnings
if(MSVC)
    target_compile_options(dsp_test PRIVATE /W4)
else()
    target_compile_options(dsp_test PRIVATE -Wall -Wextra -pedantic)
endif()

# Debug symbols
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(MSVC)
        target_compile_options(dsp_test PRIVATE /Zi)
    else()
        target_compile_options(dsp_test PRIVATE -g)
    endif()
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")