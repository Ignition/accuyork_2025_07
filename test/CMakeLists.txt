find_package(doctest CONFIG REQUIRED)
include(doctest)

add_executable(mandelbrot_tests unit_mandelbrot.cpp)
target_link_libraries(mandelbrot_tests PRIVATE doctest::doctest mandelbrot)
doctest_discover_tests(mandelbrot_tests)
