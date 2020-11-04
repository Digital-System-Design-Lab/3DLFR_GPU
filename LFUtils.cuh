#ifndef LF_UTILS_CUH_
#define LF_UTILS_CUH_ 
// Properties->CUDA C/C++->Common->generate relocatable device code=Yes

#include "enums.h"

#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <string>
#include <fcntl.h> // file open flag
#include <io.h> // file descriptor
#include <chrono>
#include <vector>
#include <queue>
#include <assert.h>
#include <mutex>
#include <conio.h> // Keyboard input 

#define LENGTH 50
#define WIDTH 5120
#define HEIGHT 2560
#define SLICE_WIDTH 256
#define OUTPUT_WIDTH 2250
#define PI 3.14159274f

const std::string g_directory = "S:/len50/5K/";
const int g_width = WIDTH;
const int g_height = HEIGHT;
const int g_length = LENGTH;
const int g_slice_width = SLICE_WIDTH;
const int g_output_width = OUTPUT_WIDTH;
const int g_slice_size = g_slice_width * g_height * 3;
const int g_LF_window_size = 3;

struct Interlaced_LF {
	int LF_number;
	ROW_COL type;
	uint8_t* odd_field;
	uint8_t* even_field;
	LF_READ_PROGRESS progress;
};

class StopWatch {
public:
	void Start();
	double Stop();
private:
	std::chrono::high_resolution_clock::time_point t0;
};

uint8_t* alloc_uint8(int size, std::string alloc_type);

void free_uint8(uint8_t* buf, std::string alloc_type);

int read_uint8(uint8_t* buf, std::string filename, const INTERLACE_FIELD& field, int size = -1);

int write_uint8(uint8_t* buf, std::string filename, int size = -1);

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

int preRendering(int z, float fov = 90.0f, float times = 270.0f);

std::vector<int> getLFUID(const int& posX, const int& posY);

void find_LF_number_BMW(int& front, int& right, int& back, int& left, const int& LFUID);

__device__ int dev_SignBitMasking(int l, int r);

__device__ int dev_Clamp(int val, int min, int max);

__device__ float dev_rad2deg(float rad);

__device__ float dev_deg2rad(float deg);

#endif