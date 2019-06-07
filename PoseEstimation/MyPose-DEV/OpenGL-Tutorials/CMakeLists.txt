SET(execName tutorial02)
CMAKE_MINIMUM_REQUIRED(VERSION 2.4)
PROJECT(${execName})

SET(EXECUTABLE_OUTPUT_PATH ".")

# OpenGL
find_package(OpenGL REQUIRED)

# GLM
find_package(glm REQUIRED)
find_package(GLUT REQUIRED)
include_directories(
    ${OPENGL_INCLUDE_DIRS} 
    ${glm_INCLUDE_DIRS}
    ${GLUT_INCLUDE_DIRS}
    /usr/include/GLFW/ # glfw
    /usr/local/cuda-10.0/extras/CUPTI/include/GL/ # glew
)

set(ALL_LIBS
	${OPENGL_LIBRARY}
    ${GLUT_LIBRARY}
	glfw
	GLEW
    glm
)

ADD_EXECUTABLE(${execName} tutorial02.cpp)
add_definitions(-std=c++1z -g)

TARGET_LINK_LIBRARIES(${execName}
    ${ALL_LIBS}
)
