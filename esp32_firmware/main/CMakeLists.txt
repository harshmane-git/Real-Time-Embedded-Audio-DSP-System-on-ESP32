idf_component_register(
    SRCS "main.c"
    INCLUDE_DIRS "."
    REQUIRES
        freertos
        ring_buffer
        audio_pipeline
        scheduler
        mic
)