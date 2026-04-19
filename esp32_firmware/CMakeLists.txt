cmake_minimum_required(VERSION 3.16)

set(EXTRA_COMPONENT_DIRS 
    ../C_implementation/low_pass
    ../C_implementation/high_pass
    ../C_implementation/Band_Pass
    ../C_implementation/biquad
)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(esp32_audio_DSP)