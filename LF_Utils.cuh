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
#define WIDTH 4096
#define HEIGHT 2048
#define SLICE_WIDTH 256
#define OUTPUT_WIDTH 9000
#define PI 3.14159274f

const size_t g_width = WIDTH;
const size_t g_height = HEIGHT;
const size_t g_length = LENGTH;
const size_t g_slice_width = SLICE_WIDTH;
const size_t g_output_width = OUTPUT_WIDTH;
const size_t g_slice_size = g_slice_width * g_height * 3;

struct SliceRange
{
	SliceRange(const FOUR_DIRECTION& dir, const int& img, const int& begin, const int& end)
	{
		direction = dir;
		image_num = img;
		range_begin = begin;
		range_end = end;
	}
	FOUR_DIRECTION direction;
	int image_num;
	int range_begin;
	int range_end;
};

typedef std::vector<SliceRange> SliceSet;

struct Interlaced_LF {
	int LF_number;
	ROW_COL type;
	uint8_t* odd_field = nullptr;
	uint8_t* even_field = nullptr;
	LF_READ_PROGRESS progress = LF_READ_PROGRESS_NOT_PREPARED;
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

int read_uint8(uint8_t* buf, std::string filename, int size = -1);

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

int preRendering(int x, int z, int dir);

void write_rendering_range();

void getLocalPosition(int& localPosX, int& localPosY, const int& curPosX, const int& curPosY);

int getLFUID(const int& posX, const int& posY);

void find_LF_number_BMW(int& front, int& right, int& back, int& left, const int& LFUID);

void constructLF_interlace();

__device__ int dev_SignBitMasking(int l, int r);

__device__ int dev_Clamp(int val, int min, int max);

__device__ float dev_rad2deg(float rad);

__device__ float dev_deg2rad(float deg);

__device__ int dev_getLFUID(const int& posX, const int& posY);

__device__ int dev_find_LF_number_BMW(const int& direction, const int& posX, const int& posY);

#endif