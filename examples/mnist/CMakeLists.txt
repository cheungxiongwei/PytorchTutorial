cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(mnist)

set(CMAKE_PREFIX_PATH "D:/libtorch-win-shared-with-deps-latest")

find_package(Torch REQUIRED)

include_directories(../../net ./)

add_executable(mnist main.cpp ../../net/lenet.hpp mnistdataset.cpp mnistdataset.h)

target_link_libraries(mnist "${TORCH_LIBRARIES}")

set_property(TARGET mnist PROPERTY CXX_STANDARD 11)


set(EXECUTABLE_OUTPUT_PATH ./bin)

# mkdir build
# cd build
# cmake -DCMAKE_BUILD_TYPE=Release -G "Visual Studio 14 2015 Win64" ..
# cmake --build .
# ./bin/Debug/mnist.exe