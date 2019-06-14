SET(execName FaceLandmarkDetector)

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(SJC)

set(CMAKE_C_COMPILER /usr/bin/gcc-7)
set(CMAKE_CXX_COMPILER /usr/bin/g++-7)

#set(Boost_USE_STATIC_LIBS OFF)
#set(BUILD_SHARED_LIBS ON)

find_package(Torch REQUIRED)
#find_package(Boost COMPONENTS system filesystem REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories("/usr/include/c++/7")

FILE(GLOB_RECURSE SRC_FILES src/*.cpp)
FILE(GLOB_RECURSE HDR_FILES include/*.h)

#MESSAGE( status " BOOST libraries" ${Boost_LIBRARIES} ) 
#MESSAGE( status ${Boost_SYSTEM_LIBRARY} )

add_executable(${execName} ${HDR_FILES} ${SRC_FILES})
target_link_libraries(${execName} ${TORCH_LIBRARIES} filesystem filesystem -rdynamic -lstdc++fs) # boost_system boost_filesystem stdc++fs)
set_property(TARGET ${execName} PROPERTY CXX_STANDARD 17)# CXX_EXTENSIONS OFF)
