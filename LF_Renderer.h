#ifndef LF_RENDERER_H_
#define LF_RENDERER_H_

#include "LRU_Cache.h"
#include "LFU_Window.h"
#include "BMW_FilePath.h"

#include <thread> // std::thread
#include <future> // std::future
#include <stdlib.h> // size_t
#include <stdint.h> // uint8_t

class LF_Renderer {
	LF_Renderer(const size_t& limit_cache_size, const size_t& limit_LF, const std::string& pixel_range_path, const std::string& LF_path);
	~LF_Renderer();

	int do_rendering(const int& newPosX, const int newPosY);

private:
	void load_slice_set(SliceSet slice_set[][100], std::string prefix);
	void loop_nbrs_h2d(LRUCache& LRU, const LFU_Window& window, SliceSet slice_set[][100], std::vector<std::pair<int, int>>& nbrPosition, cudaStream_t stream_h2d, H2D_THREAD_STATE& thread_state_h2d, const MAIN_THREAD_STATE& thread_state_main, std::mutex& mtx);
	void loop_read_disk(LFU_Window& window, const int& prevPosX, const int& prevPosY, const int& curPosX, const int& curPosY, const int& light_field_size, READ_DISK_THREAD_STATE& read_disk_thread_state, const MAIN_THREAD_STATE& main_thread_state);
	void set_rendering_range(int* localPosX, int* localPosY, int* output_width, const int& curPosX, const int& curPosY);
	void getNeighborList(std::vector<std::pair<int, int>>& nbrPosition, int curPosX, int curPosY);
	int cache_slice(LRUCache& LRU, const LFU_Window& window, SliceSet slice_set[][100], const int& posX, const int& posY);
	int cache_slice_in_background(LRUCache& LRU, const LFU_Window& window, SliceSet slice_set[][100], std::vector<std::pair<int, int>>& nbrPosition, cudaStream_t stream_h2d, H2D_THREAD_STATE& thread_state_h2d, const MAIN_THREAD_STATE& thread_state_main);

	__device__ int dev_find_pixel_location(int img, int w, int h, int width, int height, int slice_width);
	__device__ int dev_query_hashmap(int lf, int img, int slice, int width, int length, int slice_width);
	__global__ void synthesize(uint8_t* outImage, uint8_t** d_hashmap_odd, uint8_t** d_hashmap_even, int offset, int mode, int direction, int posX, int posY, int width, int height, int legnth, int slice_width, float fov = 90.0f, float times = 270.0f);

	const size_t width = 4096;
	const size_t height = 2048;
	const size_t length = 50;
	const size_t slice_width = 256;
	const size_t output_width = 9000;
	const size_t slice_size = slice_width * height * 3;

	int localPosX[4];
	int localPosY[4];
	int output_width_each_dir[4];

	int curPosX;;
	int curPosY;;
	int prevPosX;
	int prevPosY;

	std::vector<std::pair<int, int>>* nbrPosition;

	SliceSet slice_set[100][100];
	cudaStream_t stream_main, stream_h2d;

	LRUCache* LRU;
	LFU_Window* window;
	uint8_t* u_synthesized_view;

	MAIN_THREAD_STATE state_main_thread;
	H2D_THREAD_STATE state_h2d_thread;
	READ_DISK_THREAD_STATE state_read_thread;

	std::mutex mtx;
};

#endif