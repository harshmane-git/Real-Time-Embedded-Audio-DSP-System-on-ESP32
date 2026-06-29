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
# Pass the exact directory location to your source code as a macro string literal
target_compile_definitions(dsp_test PRIVATE CONFIG_FILE_PATH="D:/dsp-audio-project/config.bin")


# === IMPORTANT: Do NOT link 'm' on Windows ===
# target_link_libraries(dsp_test m)  

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

# ============================================================================
# AUTOMATICALLY COPY CONFIG.BIN TO THE EXECUTABLE DIRECTORY ON EVERY BUILD
# ============================================================================
if(TARGET dsp_test)
    add_custom_command(
        TARGET dsp_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "D:/dsp-audio-project/config.bin"
        "$<TARGET_FILE_DIR:dsp_test>/config.bin"
        COMMENT "Copying encrypted config.bin to output directory..."
    )
endif()
