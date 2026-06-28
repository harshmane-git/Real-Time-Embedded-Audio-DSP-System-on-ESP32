# file: esp32_firmware/CMakeLists.txt
cmake_minimum_required(VERSION 3.16)

# 1. Safely create a transient placeholder component directory inside the build tree
set(VIRTUAL_COMMON_DIR "${CMAKE_CURRENT_BINARY_DIR}/common")
if(NOT EXISTS "${VIRTUAL_COMMON_DIR}")
    file(MAKE_DIRECTORY "${VIRTUAL_COMMON_DIR}")
    file(WRITE "${VIRTUAL_COMMON_DIR}/CMakeLists.txt" "idf_component_register()")
endif()

# 2. Track the mathematical modules including our virtual build-space placeholder
set(EXTRA_COMPONENT_DIRS 
    "${VIRTUAL_COMMON_DIR}"
    "../C_implementation/low_pass"
    "../C_implementation/high_pass"
    "../C_implementation/Band_Pass"
    "../C_implementation/biquad"
    "../C_implementation/equalizer"
    "../C_implementation/delay"
    "../C_implementation/gain"
    "../C_implementation/limiter"
)

# Load the ESP-IDF CMake infrastructure
include($ENV{IDF_PATH}/tools/cmake/project.cmake)

project(esp32_audio_DSP)

# 3. GLOBAL TOOLCHAIN INTERVENTION: Fix C17 macro compatibility project-wide
# -include stdbool.h forces the compiler to read bool parameters natively in C17 mode
idf_build_set_property(C_COMPILE_OPTIONS "-std=gnu17" APPEND)
idf_build_set_property(C_COMPILE_OPTIONS "-include" APPEND)
idf_build_set_property(C_COMPILE_OPTIONS "stdbool.h" APPEND)
idf_build_set_property(C_COMPILE_OPTIONS "-Wno-error=unused-variable" APPEND)
idf_build_set_property(C_COMPILE_OPTIONS "-Wno-error=unused-const-variable" APPEND)

# 4. Safely inject the cross-repository type headers into the registered component context
if(TARGET __idf_common)
    target_include_directories(__idf_common INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/../C_implementation/common_types")
endif()