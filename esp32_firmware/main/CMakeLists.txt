idf_component_register(
    SRCS "main.c"
    INCLUDE_DIRS "."
    REQUIRES audio_pipeline mic amp ring_buffer common
)