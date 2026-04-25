idf_component_register(
    SRCS "high_pass.c"
    INCLUDE_DIRS "."
    REQUIRES common biquad
)