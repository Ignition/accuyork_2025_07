add_executable(mandelbrot_viewer main.cpp)
target_link_libraries(mandelbrot_viewer PRIVATE sfml::sfml mandelbrot)
target_compile_options(mandelbrot_viewer PRIVATE -march=x86-64-v3 -mtune=native)