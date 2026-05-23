cmake_minimum_required(VERSION 3.16)

set(EXTRA_COMPONENT_DIRS 
    ../C_implementation/low_pass
    ../C_implementation/high_pass
    ../C_implementation/Band_Pass
    ../C_implementation/biquad
    ../C_implementation/equalizer
    ../C_implementation/delay
    ../C_implementation/gain
)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-error=unused-variable -Wno-error=unused-const-variable")
project(esp32_audio_DSP)