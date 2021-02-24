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

#define PI 3.14159274f

struct LF_Config {
	LF_Config(std::string path_LF, std::string path_pixrange, size_t iw, size_t ih, size_t lf_len, size_t numLFs, size_t sw, double dpp)
		: path_LightField(path_LF), path_PixelRange(path_pixrange), LF_width(iw), LF_height(ih), LF_length(lf_len), num_LFs(numLFs), slice_width(sw), DPP(dpp), output_width((size_t)(360.0/dpp)), slice_size(sw * ih * 3), LF_size(iw * ih * lf_len * 3) {};
	
	const std::string path_LightField;
	const std::string path_PixelRange;
	const size_t LF_width;
	const size_t LF_height;
	const size_t LF_length;
	const size_t num_LFs;
	const size_t slice_width;
	const double DPP;
	const size_t output_width;
	const size_t slice_size;
	const size_t LF_size;
}; 

class StopWatch {
public:
	void Start();
	double Stop();
private:
	std::chrono::high_resolution_clock::time_point t0;
};

uint8_t* alloc_uint8(size_t size, std::string alloc_type);

void free_uint8(uint8_t* buf, std::string alloc_type);

int read_uint8(uint8_t* buf, std::string filename, size_t size = -1);

int write_uint8(uint8_t* buf, std::string filename, size_t size = -1);

double getEuclideanDist(int x, int y, int origX = 0, int origY = 0);

int clamp(int val, int min, int max);

double rad2deg(double rad);

double deg2rad(double deg);

void minmax(int val, int& min, int& max);

int getKey(int& posX, int& posY);

std::string IntToFormattedString(int n);

std::string FloatToFormattedString(float n);

int preRendering(int x, int z);

void write_rendering_range();

void getLocalPosition(int& localPosX, int& localPosY, const int& curPosX, const int& curPosY);

int getLFUID(const int& posX, const int& posY);

void find_LF_number_BMW(int& front, int& right, int& back, int& left, const int& LFUID);

void constructLF_interlace();

void write_bmw_fname_array(std::string path = "./BMW_FilePath.h");

int mround(int n, int m);

size_t get_devmem_freespace();

size_t get_devmem_totalpace();

void query_CudaMemory();

__device__ int dev_SignBitMasking(int l, int r);

__device__ int dev_Clamp(int val, int min, int max);

__device__ float dev_rad2deg(float rad);

__device__ float dev_deg2rad(float deg);

__device__ int dev_getLFUID(const int& posX, const int& posY);

__device__ int dev_find_LF_number_BMW(const int& direction, const int& posX, const int& posY);

#endif