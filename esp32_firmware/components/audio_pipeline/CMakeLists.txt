idf_component_register(
    SRCS "audio_pipeline.c"
    INCLUDE_DIRS "."
    REQUIRES common mic amp ring_buffer low_pass Band_Pass 
    high_pass equalizer esp_timer delay gain
    PRIV_REQUIRES esp_driver_gpio
)