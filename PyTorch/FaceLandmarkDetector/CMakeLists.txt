SET(execName FaceLandmarkDetector)

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(SJC)

find_package(Torch REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

FILE(GLOB_RECURSE SRC_FILES src/*.cpp)
FILE(GLOB_RECURSE HDR_FILES include/*.h)

add_executable(${execName} ${HDR_FILES} ${SRC_FILES})
target_link_libraries(${execName} ${TORCH_LIBRARIES})
set_property(TARGET ${execName} PROPERTY CXX_STANDARD 11)
