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

struct IO_Config {
	const size_t LF_width = 4096;
	const size_t LF_height = 2048;
	const size_t LF_length = 50;
	const size_t slice_width = 256;
	const size_t output_width = 9000;
	const size_t slice_size = slice_width * LF_height * 3;
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

int preRendering(int x, int z, int dir);

void write_rendering_range();

void getLocalPosition(int& localPosX, int& localPosY, const int& curPosX, const int& curPosY);

int getLFUID(const int& posX, const int& posY);

void find_LF_number_BMW(int& front, int& right, int& back, int& left, const int& LFUID);

void constructLF_interlace();

void write_bmw_fname_array(std::string path = "./BMW_FilePath.h");

__device__ int dev_SignBitMasking(int l, int r);

__device__ int dev_Clamp(int val, int min, int max);

__device__ float dev_rad2deg(float rad);

__device__ float dev_deg2rad(float deg);

__device__ int dev_getLFUID(const int& posX, const int& posY);

__device__ int dev_find_LF_number_BMW(const int& direction, const int& posX, const int& posY);

#endif