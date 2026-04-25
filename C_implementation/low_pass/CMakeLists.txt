idf_component_register(
    SRCS "low_pass.c"
    INCLUDE_DIRS "."
    REQUIRES common biquad
)