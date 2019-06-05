// Include standard headers
#include <stdio.h>
#include <stdlib.h>

// Include GLEW. Always include it before gl.h and glfw3.h, since it's a bit magic.
#include <GL/glew.h>
// Include GLFW: let GLFW handle the window and the keyboard
#include <GLFW/glfw3.h>
// Include GLM:library for 3D mathematics
#include <glm/glm.hpp>

int main(int argc, char **argv) {
    // Initialise GLFW
    glewExperimental = true; // Needed for core profile
    if ( !glfwInit()) {
        fprintf( stderr, "Failed to initialize GLFW\n" );
        return -1;
    }

    glfwWindowHint(GLFW_SAMPLES, 4); // 4x antialiasing
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);; // OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // for MacOS
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // We don't want the old OpenGL

    // Open a window and create its OpenGL context
    GLFWwindow *window;  // (In the accompanying source code, this variable is global for simplicity)
    window = glfwCreateWindow(1024, 768, "Tutorial 01", NULL, NULL);
    if (window == NULL) {
        fprintf(stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n");
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window); // Initialize GLEW
    glewExperimental = true; // Needed in core profile
    if (glewInit() != GLEW_OK) {
        fprintf( stderr, "Failed to initialize GLEW\n" );
        return -1;
    }

    // Ensure we can capture the excape key being pressed below
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
	
    // Dark blue background
	glClearColor(0.0f, 0.0f, 0.4f, 0.0f);
    
    do {
        // Clear the screen.
        glClear(GL_COLOR_BUFFER_BIT);

        // Draw nothing

        // Swap buffers
        glfwSwapBuffers(window);
        glfwPollEvents();
    } 
    while(glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(window) == 0);

    glfwTerminate();

    return 0;
}
