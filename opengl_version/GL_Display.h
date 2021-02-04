#ifndef GL_DISPLAY_
#define GL_DISPLAY_ 

#include <Windows.h>
#include <helper_gl.h>
#include <GL/freeglut.h>
#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <rendercheck_gl.h>

#include "LF_Utils.cuh"

extern GLuint _GL_pbo;
extern GLuint _GL_texture;
extern std::string _GL_window_name;
extern int _GL_width;
extern int _GL_height;
extern int _GL_channel;
extern int _GL_window_width;
extern int _GL_window_height;
extern int _GL_pitch;
extern unsigned short _GL_channel_type;
extern unsigned short _GL_channel_sequence;
extern unsigned short _GL_image_depth;
extern int _GL_waitKeyCalled;
extern int _GL_timeDelay;
extern int _GL_pressed_key;

void initDisplay(int width, int height, int channels);
void display();

void _createTexture(GLuint* textureId, int width, int height, int channels);
void _createPBO(GLuint* pbo);
void _runCUDA(uint8_t* _CUDA_pixelArray);
void _keyboardFunction(unsigned char key, int x, int y); // need to register a callback function
void _mouseWheelFunc(int wheel, int direction, int x, int y); // need to register a callback function
void _timerFunction(int value); // need to register a callback function
void _display(); // need to register a callback function
void _closingFuntion(); // need to register a callback function
void _initCUDA();
void _initGL();
#endif