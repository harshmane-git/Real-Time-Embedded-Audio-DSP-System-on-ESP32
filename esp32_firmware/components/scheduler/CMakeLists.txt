idf_component_register(
    SRCS "scheduler.c"
    INCLUDE_DIRS "."
    REQUIRES ring_buffer amp
)