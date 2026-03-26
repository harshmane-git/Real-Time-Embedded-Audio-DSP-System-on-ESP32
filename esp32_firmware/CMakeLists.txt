cmake_minimum_required(VERSION 3.22)

set(EXTRA_COMPONENT_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/../C_implementation"
)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(esp32_audio_dsp)
