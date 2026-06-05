idf_component_register(
    SRCS "equalizer.c"
    INCLUDE_DIRS "."
    REQUIRES common low_pass Band_Pass high_pass
)