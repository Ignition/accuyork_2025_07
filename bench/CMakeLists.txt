find_package(benchmark REQUIRED)

add_executable(bench bench.cpp)
target_link_libraries(bench PRIVATE benchmark::benchmark mandelbrot)
#target_compile_options(bench PRIVATE -march=native)