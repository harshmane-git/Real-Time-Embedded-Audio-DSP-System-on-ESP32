cmake_minimum_required(VERSION 3.10)

project(UnitTest)

set(CMAKE_C_STANDARD 99)

# =====================================================
# Generator Unit Test
# =====================================================

add_executable(test_generator
    test_generator.c
    config_generator.c
)

# =====================================================
# Loader Unit Test
# =====================================================

add_executable(test_loader
    test_loader.c
    config_loader.c
)

target_link_libraries(test_loader m)

# =====================================================
# End-to-End Unit Test
# =====================================================

add_executable(test_end_to_end
    test_end_to_end.c
    config_generator.c
    config_loader.c
)

target_link_libraries(test_end_to_end m)
