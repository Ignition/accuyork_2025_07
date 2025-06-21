cmake_minimum_required(VERSION 3.31)
project(demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
    add_compile_options(-stdlib=libc++)
    add_link_options(-stdlib=libc++)
endif()

add_library(mandelbrot INTERFACE)
target_sources(mandelbrot
        INTERFACE
        FILE_SET HEADERS FILES
        include/mandelbrot/mandelbrot.hpp
        include/mandelbrot/v1.hpp
        include/mandelbrot/v2.hpp
        include/mandelbrot/v3.hpp
        include/mandelbrot/v4.hpp
        include/mandelbrot/v5.hpp
        include/mandelbrot/v6.hpp
        include/mandelbrot/v7.hpp
        include/mandelbrot/v7.hpp
)

target_include_directories(mandelbrot INTERFACE include)

# dependency via conan
find_package(xsimd REQUIRED)
find_package(libdispatch REQUIRED)
find_package(TBB REQUIRED)

# dependency via CPM
include(cmake/CPM.cmake)
CPMAddPackage(
        NAME stdexec
        GITHUB_REPOSITORY NVIDIA/stdexec
        GIT_TAG main
)

target_link_libraries(mandelbrot INTERFACE STDEXEC::stdexec xsimd libdispatch::libdispatch onetbb::onetbb)
target_link_options(mandelbrot INTERFACE -fblocks)

install(TARGETS mandelbrot DESTINATION "."
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
)

#include(CTest)
#if (BUILD_TESTING)
#    add_subdirectory(test)
#endif ()
add_subdirectory(bench)
