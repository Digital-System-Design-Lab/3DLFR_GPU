#ifndef LF_UTILS_H_
#define LF_UTILS_H_

#include <iostream>
#include <string>
#include <chrono>
#include <vector>
#include <assert.h>
#include <mutex>
#include <queue>
#include <conio.h> // Keyboard input 

#define LENGTH 50
#define WIDTH 5120
#define HEIGHT 2560
#define SLICE_WIDTH 256
#define OUTPUT_WIDTH 2250

const std::string g_directory = "S:/len50/5K/";
const int g_width = WIDTH;
const int g_height = HEIGHT;
const int g_length = LENGTH;
const int g_slice_width = SLICE_WIDTH;
const int g_output_width = OUTPUT_WIDTH;
const int g_slice_size = g_slice_width * g_height * 3;
const int g_LF_window_size = 3;

#define PI 3.14159274f

enum MAIN_THREAD_STATE {
	MAIN_THREAD_TERMINATED = -3,
	MAIN_THREAD_INIT = -2,
	MAIN_THREAD_WAIT = -1,
	MAIN_THREAD_H2D = 0,
	MAIN_THREAD_RENDERING = 1,
	MAIN_THREAD_D2H = 2,
	MAIN_THREAD_COMPLETE = 3
};

enum H2D_THREAD_STATE {
	H2D_THREAD_TERMINATED = -4,
	H2D_THREAD_INIT = -3,
	H2D_THREAD_INTERRUPTED = -2,
	H2D_THREAD_WAIT = -1,
	H2D_THREAD_RUNNING = 0
};

enum READ_DISK_THREAD_STATE {
	READ_DISK_THREAD_TERMINATED = -2,
	READ_DISK_THREAD_INIT = -1,
	READ_DISK_THREAD_CURRENT_LF_READING = 0,
	READ_DISK_THREAD_CURRENT_LF_READ_COMPLETE = 1,
	READ_DISK_THREAD_NEIGHBOR_LF_READING = 2,
	READ_DISK_THREAD_NEIGHBOR_LF_READ_COMPLETE = 3,
};

enum LF_READ_PROGRESS {
	LF_READ_PROGRESS_NOT_PREPARED = -1,
	LF_READ_PROGRESS_ODD_FIELD_PREPARED = 0,
	LF_READ_PROGRESS_EVEN_FIELD_PREPARED = 1
};

struct Interlaced_LF {
	int LF_number;
	uint8_t* odd_field;
	uint8_t* even_field;
	// uint8_t* full_field;
	LF_READ_PROGRESS progress;
};

class StopWatch {
public:
	void Start();
	double Stop();
private:
	std::chrono::high_resolution_clock::time_point t0;
};

double getEuclideanDist(int x, int y, int origX = 0, int origY = 0);

int clamp(int val, int min, int max);

double rad2deg(double rad);

double deg2rad(double deg);

void minmax(int val, int& min, int& max);

int getKey(int& posX, int& posY);

std::string IntToFormattedString(int n);
std::string FloatToFormattedString(float n);

double differentiation(double prev, double cur, double timespan);
std::pair<double, double> deadReckoning(std::pair<double, double> a_2, std::pair<double, double> a_1, std::pair<double, double> a0, double framerate, int f);
std::vector<std::pair<double, std::pair<int, int>>> doDeadReckoning(double prevprevPosX, double prevprevPosY, double prevPosX, double prevPosY, double curPosX, double curPosY, double framerate, int pred);
int find_slice_from_LF(const int& img, const int& slice, bool interlaced = false);
Interlaced_LF* get_LF_from_Window(std::vector<Interlaced_LF>& window, const int& LF_number);


#endif