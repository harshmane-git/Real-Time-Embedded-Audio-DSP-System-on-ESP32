idf_component_register(
    SRCS "audio_pipeline.c"
    INCLUDE_DIRS "."
    REQUIRES
        mic
        amp
)