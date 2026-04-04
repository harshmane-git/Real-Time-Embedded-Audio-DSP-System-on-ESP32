idf_component_register(
    SRCS "mic.c"
    INCLUDE_DIRS "."
    REQUIRES
        ring_buffer
        driver
        freertos
)