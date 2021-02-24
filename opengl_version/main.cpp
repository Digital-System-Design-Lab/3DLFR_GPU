#include "LF_Renderer.cuh"
#include <Windows.h>
#include <helper_gl.h>
#include <GL/freeglut.h>
#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <rendercheck_gl.h>

#define RESOLUTION 8
#if RESOLUTION == 1
#define WIDTH 1024
#define HEIGHT 512
#define PATH_LF "S:/BMW_1K/"
#define PATH_PIXEL_RANGE "S:/PixelRange_1K/"
#elif RESOLUTION == 2
#define WIDTH 2048
#define HEIGHT 1024
#define PATH_LF "S:/BMW_2K/"
#define PATH_PIXEL_RANGE "S:/PixelRange_2K/"
#elif RESOLUTION == 4
#define WIDTH 4096
#define HEIGHT 2048
#define PATH_LF "S:/BMW_4K/"
#define PATH_PIXEL_RANGE "S:/PixelRange_4K/"
#elif RESOLUTION == 8
#define WIDTH 7680
#define HEIGHT 4320
#define PATH_LF "S:/BMW_8K/"
#define PATH_PIXEL_RANGE "S:/PixelRange_8K/"
#endif

GLuint _GL_pbo = NULL;
GLuint _GL_texture = NULL;
std::string _GL_window_name;
int _GL_width;
int _GL_height;
int _GL_channel;
int _GL_window_width;
int _GL_window_height;
int _GL_pitch;
unsigned short _GL_channel_type;
unsigned short _GL_channel_sequence;
unsigned short _GL_image_depth;
int _GL_waitKeyCalled = 0;
int _GL_timeDelay = 0;
int _GL_pressed_key = -1;


int curPosX = 201;
int curPosY = 298;
int LF_length = 50;
int num_LFs = 734;
double dpp = 0.04;

LF_Renderer renderer(PATH_LF, PATH_PIXEL_RANGE, WIDTH, HEIGHT, LF_length, num_LFs, dpp, curPosX, curPosY, false);

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

int main()
{
#if 1
	initDisplay((int)(360.0 / dpp), HEIGHT, 3);
	display();
#else
	double elapsed_time[99];
	for (int x = curPosX; x < 300; x++)
	{
		sw.Start();
		uint8_t* synthesized_view = renderer.do_rendering(x, curPosY);
		double ms = sw.Stop();
		printf("%f ms (%.3f Hz)\n", ms, (1 / ms * 1000));
		elapsed_time[(x % 100) - 1] = ms;

		FILE* fp = fopen(("./result/view/[9000x2048] " + std::to_string(x) + "_" + std::to_string(curPosY) + ".bgr").c_str(), "wb");
		fwrite(synthesized_view, 1, 9000 * 2048 * 3, fp);
		fclose(fp);
	}

	/* log elapsed time */
	FILE* file_time_log = fopen(("./experiments/elapsed/" + std::to_string(renderer.get_configuration().slice_width) + "_elapsed_time.log").c_str(), "w");
	for (int i = 0; i < 99; i++) {
		fprintf(file_time_log, "%f\n", elapsed_time[i]);
	}
	fclose(file_time_log);
	/* log elapsed time - end*/
#endif
	renderer.terminate();

	return 0;
}

void _timerFunction(int value)
{
	if (value == 1)
		glutLeaveMainLoop();
	glutPostRedisplay();
}

void _createPBO(GLuint* pbo)
{
	if (pbo)
	{
		int data_size = _GL_width * _GL_height * _GL_channel * sizeof(GLubyte);

		glGenBuffers(1, pbo);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo);
		glBufferData(GL_PIXEL_UNPACK_BUFFER, data_size, NULL, GL_DYNAMIC_COPY);
		cudaGLRegisterBufferObject(*pbo);
	}
}

void _mouseWheelFunc(int wheel, int direction, int x, int y)
{
	// if (!full)
	{
		if (direction > 0)
		{
			_GL_window_width /= 0.97f;
			_GL_window_height /= 0.97f;
		}
		else
		{
			_GL_window_width *= 0.97f;
			_GL_window_height *= 0.97f;
		}
		glutReshapeWindow(_GL_window_width, _GL_window_height);

		glutPostRedisplay();
	}
}

void _keyboardFunction(unsigned char key, int x, int y)
{
	_GL_pressed_key = key;
	int mod = glutGetModifiers();

	switch (key)
	{
	case 'x': {			curPosY--; } break;
	case 'c': {	curPosX++;	curPosY--; } break;
	case 'd': {	curPosX++;			} break;
	case 'e': {	curPosX++;	curPosY++;	} break;
	case 'w': {			curPosY++; } break;
	case 'q': {	curPosX--; curPosY++; } break;
	case 'a': {	curPosX--; } break;
	case 'z': {	curPosX--; curPosY--; } break;
	case 27: {	
		printf("Terminate\n"); 
		renderer.terminate();
		exit(0);	
	}
	default: break;
	}
	curPosX = clamp(curPosX, (curPosX / 100) * 100 + 1, (curPosX / 100) * 100 + 99);
	curPosY = clamp(curPosY, (curPosY / 100) * 100 + 1, (curPosY / 100) * 100 + 99);
	glutPostRedisplay();
}

void _closingFuntion()
{
	cudaGLUnregisterBufferObject(_GL_pbo);
	glDeleteTextures(1, &_GL_texture);
	glDeleteBuffers(1, &_GL_pbo);
	glDisable(GL_TEXTURE_2D);
}

void _initGL()
{
	glutInit(&__argc, __argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);
	unsigned int desk_w = 0, desk_h = 0;

	desk_w = glutGet(GLUT_SCREEN_WIDTH);
	desk_h = glutGet(GLUT_SCREEN_HEIGHT);

	if (_GL_width > desk_w || _GL_height > desk_h)
		glutInitWindowPosition(0, 0);
	else
		glutInitWindowPosition((desk_w - _GL_width) / 2, (desk_h - _GL_height) / 2);
	glutInitWindowSize(_GL_window_width, _GL_window_height);
	glutCreateWindow(_GL_window_name.c_str());
	glutDisplayFunc(_display);
	glutKeyboardFunc(_keyboardFunction);
	glutMouseWheelFunc(_mouseWheelFunc);
	glutCloseFunc(_closingFuntion);

	glewInit();
	glViewport(0, 0, _GL_width, _GL_height);
	glDisable(GL_DEPTH_TEST);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
}

void _createTexture(GLuint* tex, int width, int height, int channels)
{
	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, tex);
	glBindTexture(GL_TEXTURE_2D, *tex);

	glTexImage2D(GL_TEXTURE_2D, 0, _GL_channel_type, width, height, 0, _GL_channel_sequence, _GL_image_depth, NULL);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

void _initCUDA()
{
	cudaGLSetGLDevice(0);
	_createTexture(&_GL_texture, _GL_width, _GL_height, _GL_channel);
	_createPBO(&_GL_pbo);
}

void initDisplay(int width, int height, int channels)
{

	_GL_width = width;
	_GL_height = height;
	_GL_channel = channels;
	_GL_pitch = width * _GL_channel;
	_GL_window_width = width / 4;
	_GL_window_height = height / 4;
	_GL_window_name = std::to_string(-1) + "," + std::to_string(-1);

	switch (_GL_channel)
	{
	case 1:
		_GL_channel_type = GL_LUMINANCE;
		_GL_channel_sequence = GL_LUMINANCE;
		_GL_image_depth = GL_UNSIGNED_BYTE;

	case 3:
		_GL_channel_type = GL_RGB;
		_GL_channel_sequence = GL_RGB;
		_GL_image_depth = GL_UNSIGNED_BYTE;
		break;

	case 4:
		_GL_channel_type = GL_RGBA;
		_GL_channel_sequence = GL_RGBA;
		_GL_image_depth = GL_UNSIGNED_BYTE;
		break;
	default:
		exit(1);
	}

	_initGL();
	_initCUDA();
}

void _runCUDA(uint8_t* _CUDA_pixelArray)
{
	void* dptr = NULL;

	cudaGLMapBufferObject((void**)&dptr, _GL_pbo);
	cudaMemcpy2D(dptr, _GL_pitch, _CUDA_pixelArray, _GL_pitch, _GL_pitch, _GL_height, cudaMemcpyDeviceToDevice);
	cudaGLUnmapBufferObject(_GL_pbo);
}

void _display()
{
	StopWatch latencyE2E;
	latencyE2E.Start();
	
	_runCUDA(renderer.do_rendering(curPosX, curPosY));

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _GL_pbo);
	glBindTexture(GL_TEXTURE_2D, _GL_texture);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _GL_width, _GL_height, _GL_channel_sequence, _GL_image_depth, NULL)             ;
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 1.0f);    glVertex2f(0.0f, 0.0f);
	glTexCoord2f(0.0f, 0.0f);    glVertex2f(0.0f, 1.0f);
	glTexCoord2f(1.0f, 0.0f);    glVertex2f(1.0f, 1.0f);
	glTexCoord2f(1.0f, 1.0f);    glVertex2f(1.0f, 0.0f);
	glEnd();
	glutSwapBuffers();

	double ms = latencyE2E.Stop();
	std::string windowTitle = "Position: (" + std::to_string(curPosX) + "," + std::to_string(curPosY) + ") " + std::to_string(ms) + "ms, " + std::to_string(1 / ms * 1000) + "FPS";
	glutSetWindowTitle(windowTitle.c_str());

#if 0
	/* log elapsed time */
	FILE* file_elapsedT = fopen(("./experiments/elapsed/prefetch/" + std::to_string(renderer.io_config.slice_width) + "_time_prefetch.log").c_str(), "a");
	if(curPosX != 201)
		fprintf(file_elapsedT, "%d\t%d\t%f\n", curPosX, curPosY, ms);
	fclose(file_elapsedT);
	/* log hit rate - end*/
#endif
	if (curPosX == 299) {
		renderer.terminate(); 
		exit(1);
	}

// 	printf("%f ms (%.3f Hz)\n", ms, (1 / ms * 1000));
}

void display()
{
	glutDisplayFunc(_display);
	glutMainLoop();
}
