# esp32_firmware/main/CMakeLists.txt
idf_component_register(
    SRCS 
        "main.c"
        "../src/pipeline.c"
        "../src/switch_control.c"
        "../src/mic.c"
        "../src/amp.c"
        "../src/ring_buffer.c"
        "../src/audio_pipeline.c"
    INCLUDE_DIRS 
        "../inc"
    REQUIRES 
        low_pass high_pass Band_Pass biquad equalizer gain delay limiter
        esp_driver_i2s esp_timer esp_driver_gpio
)