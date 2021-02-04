#include "GL_Display.h"

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
	case 102:
		// glutFullScreenToggle();
		// full = 1 - full;
		break;

	default:
		// glutLeaveMainLoop();
		break;
	} 
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
