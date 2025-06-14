cmake_minimum_required(VERSION 3.31)
project(demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(mandelbrot INTERFACE)
target_sources(mandelbrot
        INTERFACE
        FILE_SET HEADERS FILES
        include/mandelbrot/mandelbrot.hpp
        include/mandelbrot/v1.hpp
        include/mandelbrot/v2.hpp
)
target_include_directories(mandelbrot INTERFACE include)
find_package(xsimd REQUIRED)
target_link_libraries(mandelbrot INTERFACE xsimd)

install(TARGETS mandelbrot DESTINATION "."
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
)

include(CTest)
if (BUILD_TESTING)
    add_subdirectory(test)
endif ()
add_subdirectory(bench)
