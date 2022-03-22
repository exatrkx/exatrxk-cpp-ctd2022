cmake_minimum_required(VERSION 3.17)
list(APPEND CMAKE_MODULE_PATH /workspace/install/lib/cmake)


project(TritonExample VERSION 0.1.0 LANGUAGES C CXX)

find_package(TritonCore CONFIG REQUIRED)
find_package(TritonCommon CONFIG REQUIRED)
find_package(TritonClient CONFIG REQUIRED)

add_subdirectory(src)