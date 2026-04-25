idf_component_register(
    SRCS "audio_pipeline.c"
    INCLUDE_DIRS "."
    REQUIRES common mic amp ring_buffer
)