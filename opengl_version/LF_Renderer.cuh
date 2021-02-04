#ifndef LF_RENDERER_H_
#define LF_RENDERER_H_

#include "LRU_Cache.h"
#include "LFU_Window.h"
#include <thread> // std::thread
#include <stdlib.h> // size_t
#include <stdint.h> // uint8_t

__device__ int dev_find_pixel_location(int img, int w, int h, int width, int height, int slice_width);
__device__ int dev_query_hashmap(int lf, int img, int slice, int width, int length, int slice_width);
__global__ void synthesize(uint8_t* outImage, uint8_t** d_hashmap_odd, uint8_t** d_hashmap_even, int offset, int mode, int direction, int posX, int posY, int localPosX, int localPosY, int width, int height, int legnth, int slice_width, float fov = 90.0f, float times = 270.0f);

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

class LF_Renderer {
public:
	LF_Renderer(const std::string& pixel_range_path, const std::string& LF_path, const int& initPosX = 150, const int& initPosY = 150, bool use_window = true, const size_t& limit_LF = 734);
	~LF_Renderer();

	uint8_t* do_rendering(int& newPosX, int& newPosY);
	IO_Config get_configuration();
	void terminate();
	uint8_t* synthesized_view;

private:
	void load_slice_set(SliceSet slice_set[][100], std::string prefix);
	void loop_nbrs_h2d(LRU_Cache& LRU, const LFU_Window& window, SliceSet slice_set[][100], std::vector<std::pair<int, int>>& nbrPosition, cudaStream_t stream_h2d, const MAIN_THREAD_STATE& thread_state_main, std::mutex& mtx);
	void loop_read_disk(LFU_Window& window, const int& curPosX, const int& curPosY, const int& light_field_size, const MAIN_THREAD_STATE& main_thread_state);
	void set_rendering_params(int* localPosX, int* localPosY, int* output_width, const int& curPosX, const int& curPosY);
	void getNeighborList(std::vector<std::pair<int, int>>& nbrPosition, const int& curPosX, const int& curPosY);
	std::pair<int, int> cache_slice(LRU_Cache& LRU, const LFU_Window& window, SliceSet slice_set[][100], const int& posX, const int& posY);
	int cache_slice_in_background(LRU_Cache& LRU, const LFU_Window& window, SliceSet slice_set[][100], std::vector<std::pair<int, int>>& nbrPosition, cudaStream_t stream_h2d, const MAIN_THREAD_STATE& thread_state_main);
	int find_slice_from_LF(const int& img, const int& slice);

	IO_Config io_config;

	int localPosX[4];
	int localPosY[4];
	int output_width_each_dir[4];

	int curPosX;
	int curPosY;
	int prevPosX;
	int prevPosY;

	std::vector<std::pair<int, int>> nbrPosition;

	SliceSet slice_set[100][100];
	cudaStream_t stream_main, stream_h2d;

	LRU_Cache* LRU;
	LFU_Window* window;
	
	MAIN_THREAD_STATE state_main_thread;
	H2D_THREAD_STATE state_h2d_thread;
	DISK_READ_THREAD_STATE state_disk_read_thread;

	std::mutex mtx;
	std::vector<std::thread> workers;
	bool use_window;
};

#endif