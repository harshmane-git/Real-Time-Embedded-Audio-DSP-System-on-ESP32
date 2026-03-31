idf_component_register(
    SRCS "main.c"
    INCLUDE_DIRS "."
    REQUIRES ring_buffer mic amp scheduler
)