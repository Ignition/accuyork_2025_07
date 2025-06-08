cmake_minimum_required(VERSION 3.15)
project(hello CXX)

#find_package(RocksDB REQUIRED)

add_executable(hello src/hello.cpp src/main.cpp)
#target_link_libraries(hello PRIVATE RocksDB::rocksdb)

install(TARGETS hello DESTINATION "."
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        )
