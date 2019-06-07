#include <GLFW/glfw3.h>
#include <iostream>

int main(void) {
    // Initialize GLFW and create a GLFW window object (640 x 480):
    GLFWwindow* window;
    if (!glfwInit()) exit (EXIT_FAILURE);

    window = glfwCreateWindow(640, 480, "Chapter 1: Simple GLFW Example", NULL, NULL);
    if (!window) {
        glfwTerminate();
        exit (EXIT_FAILURE);
    }
    glfwMakeContextCurrent(window);

    // Define a loop that terminates when the window is closed:
    while (!glfwWindowShouldClose(window)) {
        // Set up the viewport (using the width and height of the window) and clear the screen color buffer:
        float ratio;
        int width, height;

        glfwGetFramebufferSize(window, &width, &height);
        ratio = (float) width / (float) height;

        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT);

        // Set up the camera matrix
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(-ratio, ratio, -1.f, 1.f, 1.f, -1.f);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        // Draw a rotating triangle and set a different color (red, green, and blue channels) for
        // each vertex (x, y, and z) of the triangle. The first line rotates the triangle over time:
        glRotatef((float)glfwGetTime() * 50.f, 0.f, 0.f, 1.f);
        glBegin(GL_TRIANGLES);
        glColor3f(1.f, 0.f, 0.f);
        glVertex3f(-0.6f, -0.4f, 0.f);
        glColor3f(0.f, 1.f, 0.f);
        glVertex3f(0.6f, -0.4f, 0.f);
        glColor3f(0.f, 0.f, 1.f);
        glVertex3f(0.f, 0.6f, 0.f);
        glEnd();

        // Swap the front and back buffers (GLFW uses double buffering) to update the screen
        // and process all pending events:
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Release the memory and terminate the GLFW library. Then, exit the application:
    glfwDestroyWindow(window);
    glfwTerminate();
    exit (EXIT_SUCCESS);
}