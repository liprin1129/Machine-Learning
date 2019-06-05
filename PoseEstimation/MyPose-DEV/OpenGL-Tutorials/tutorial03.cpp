#include <glut.h>

#include <math.h>

#include <iostream>


using namespace std;


float speed = 0;

void Display() {

    glClearColor(0, 0, 0, 1);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_DEPTH_TEST);

    glClearDepth(1.0f);

    glLoadIdentity();

    gluLookAt(0, 5, -15, 0, 0, 1, 0, 1, 0);

    glPushMatrix(); {

        glColor3f(1, 0.5, 0);

        glTranslatef(-4, 0, -5);

        glRotatef(speed, 0, 1, 0);

        glutWireTeapot(1);

    }

    glPopMatrix();

    glColor3f(1, 1, 0);

    // 큐브

    glBegin(GL_LINE_LOOP);

    glPushMatrix(); {    // 제 1면

        glVertex3f(1, 0, 4);

        glVertex3f(0, 0, 4);

        glVertex3f(0, 1, 4);

        glVertex3f(1, 1, 4);

    }

    glPopMatrix();

    glPushMatrix(); {    // 제 2면

        glVertex3f(1, 0, 4);

        glVertex3f(1, 0, 5);

        glVertex3f(1, 1, 5);

        glVertex3f(1, 1, 4);

    }

    glPopMatrix(); 


    glEnd();

    glBegin(GL_LINE_LOOP);

    // 삼각뿔

    glPushMatrix(); {    // 제 1면

        glVertex3f(2, 0, 0);

        glVertex3f(4, 0, 0);

        glVertex3f(3, 1, 0);

    }

    glPopMatrix();

    glPushMatrix(); {    // 제 2면

        glVertex3f(3, 0, 1);

        glVertex3f(4, 0, 0);

        glVertex3f(3, 1, 0);

    }

    glPopMatrix();

    glEnd();

    // 구 만들기

    float radius = 1;

    float radian = 3.141592 / 180;

    float maxAnlge = 360;

    for (int i = 0; i <= maxAnlge; i++) {

        glRotatef(i, 0, 1, 0);

        glBegin(GL_LINE_STRIP);

        glColor3f(0.8, 0.6, 0);

        glPushMatrix(); {    // 원 그리기

            for (int i = 0; i <= maxAnlge; i++) {

                radian += i;

                glVertex3f(cos(radian) * radius, sin(radian) * radius, 0); 

            }

        }

        glPopMatrix();

        glEnd();

    }

    glFlush();

}

void reShape(int w, int h) {

    glViewport(0, 0, w, h);

    glMatrixMode(GL_PROJECTION);

    glLoadIdentity();

    glFrustum(-1, 1, -1, 1, 1.5, 50);

    glMatrixMode(GL_MODELVIEW);

}

void idle() {

    speed += 0.5f;

    Display();

}

int main(int argc, char** argv){

    glutInit(&argc, argv);
    
    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);

    glShadeModel(GL_FLAT);

    glutInitWindowSize(1000, 1000);// 창크기

    glutCreateWindow("hello world");// 창생성

    glutDisplayFunc(Display);

    glutReshapeFunc(reShape);

    glutIdleFunc(idle);

    glutMainLoop();

    return 0;
}
