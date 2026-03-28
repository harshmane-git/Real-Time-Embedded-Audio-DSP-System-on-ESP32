idf_component_register(
    SRCS 
        "test_app_main.c"
        "test_ring_buffer.c"
    INCLUDE_DIRS "."
    REQUIRES 
        ring_buffer
        freertos
)