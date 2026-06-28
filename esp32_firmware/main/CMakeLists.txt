# file: esp32_firmware/main/CMakeLists.txt

# Group the localized implementation files from the new layout
set(srcs
    "main.c"
    "../src/mic.c"
    "../src/amp.c"
    "../src/ring_buffer.c"
    "../src/audio_pipeline.c"
)

# Register the main target block with explicitly listed peripheral requirements
idf_component_register(
    SRCS ${srcs}
    INCLUDE_DIRS "."              # Only main.c's directory is public
    PRIV_INCLUDE_DIRS "../inc"    # Keeps application headers private to your logic
    REQUIRES 
        driver
        esp_driver_gpio    # Native GPIO driver context
        esp_driver_i2s     # CRITICAL: Native ESP-IDF v5/v6 standard I2S header resolver
        freertos           # Multi-tasking support context
        esp_timer
        low_pass           # Sibling DSP mathematical components
        high_pass
        Band_Pass
        biquad
        equalizer
        delay
        gain
        limiter
)