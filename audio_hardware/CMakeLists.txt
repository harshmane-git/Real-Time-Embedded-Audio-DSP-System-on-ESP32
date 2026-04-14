idf_component_register(
    SRCS "main.c"
         "mic.c"
         "amp.c"
         "ring_buffer.c"
         "audio_pipeline.c"
    INCLUDE_DIRS "."
    REQUIRES driver
)