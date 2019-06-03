CMAKE_MINIMUM_REQUIRED(VERSION 2.4)
PROJECT(Encoder)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

#include_directories("${PROJECT_SOURCE_DIR}")

add_executable(Encoder encoder.cpp)

target_link_libraries(Encoder Eigen3::Eigen)