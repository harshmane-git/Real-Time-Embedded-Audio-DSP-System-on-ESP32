idf_component_register(
    SRCS "amp.c"
    INCLUDE_DIRS "."
    REQUIRES
        ring_buffer
        driver
        freertos
)