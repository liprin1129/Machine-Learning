SET(execName FaceLandmarksViewer)

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(SJC)

set(CMAKE_C_COMPILER /usr/bin/gcc-7)
set(CMAKE_CXX_COMPILER /usr/bin/g++-7)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ZED 2 REQUIRED)
find_package(CUDA ${ZED_CUDA_VERSION} EXACT REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
#include_directories("/usr/include/c++/7")
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${ZED_INCLUDE_DIRS})

link_directories(${ZED_LIBRARY_DIR})
link_directories(${OpenCV_LIBRARY_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

FILE(GLOB_RECURSE SRC_FILES src/*.cpp)
FILE(GLOB_RECURSE HDR_FILES include/*.h)

add_executable(${execName} ${HDR_FILES} ${SRC_FILES})

target_link_libraries(
    ${execName} 
    ${TORCH_LIBRARIES} 
    ${OpenCV_LIBS} 
    ${ZED_LIBRARIES} 
    ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} ${CUDA_NPP_LIBRARIES_ZED}
    -lstdc++fs
)

set_property(TARGET ${execName} PROPERTY CXX_STANDARD 17)# CXX_EXTENSIONS OFF)


# cmake -DCMAKE_PREFIX_PATH=/usr/local/lib/python3.6/dist-packages/torch ..
