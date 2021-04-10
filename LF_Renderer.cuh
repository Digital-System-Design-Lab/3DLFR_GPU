#ifndef LF_RENDERER_H_
#define LF_RENDERER_H_

#include "LRU_Cache.h"
#include "LFU_Window.h"
#include <thread> // std::thread

__device__ int dev_find_pixel_location(int img, int w, int h, int width, int height, int slice_width);
__device__ int dev_query_hashmap(int lf, int img, int slice, int width, int length, int slice_width);
__global__ void synthesize(uint8_t* outImage, uint8_t** d_hashmap_odd, uint8_t** d_hashmap_even, int offset, int mode, int direction, int posX, int posY, int localPosX, int localPosY, float dpp, int width, int height, int legnth, int slice_width, float fov = 90.0f, float times = 270.0f);

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
	LF_Renderer(const std::string& path_LF, const std::string& path_pixelrange, const size_t& iw, const size_t& ih, const size_t& lf_length, const size_t& num_LFs, const double& dpp, const int& stride, const int& initPosX = 150, const int& initPosY = 150, bool use_window = true);
	~LF_Renderer();

	uint8_t* do_rendering(int& newPosX, int& newPosY);
	void terminate();
	uint8_t* synthesized_view;

	void fill_cache();

private:
	void load_slice_set(SliceSet slice_set[][100], std::string prefix);
	void loop_background_prefetch(LRU_Cache& LRU, const LFU_Window& window, SliceSet slice_set[][100], std::vector<std::pair<int, int>>& nbrPosition, cudaStream_t stream_h2d, std::mutex& mtx);
	void loop_background_LF_read(LFU_Window& window, const int& curPosX, const int& curPosY, const int& light_field_size, const MAIN_THREAD_STATE& main_thread_state);
	void set_rendering_params(int* localPosX, int* localPosY, int* output_width, const int& curPosX, const int& curPosY);
	void predictFuturePosition();
	std::pair<int, int> cache_slice(LRU_Cache& LRU, const LFU_Window& window, SliceSet slice_set[][100], const int& posX, const int& posY);
	int cache_slice_in_background(LRU_Cache& LRU, const LFU_Window& window, SliceSet slice_set[][100], std::vector<std::pair<int, int>>& nbrPosition, cudaStream_t stream_h2d);
	size_t find_slice_from_LF(const int& img, const int& slice);
	void generate_LFConfig(const std::string& path_LF, const std::string& path_pixelrange, const size_t& iw, const size_t& ih, const size_t& lf_len, const size_t& numLFs, const double& dpp);

	std::pair<double, double> deadReckoning(std::pair<double, double> a_2, std::pair<double, double> a_1, std::pair<double, double> a0, double framerate, int f);
	void do_dead_reckoning(std::vector<std::pair<int, int>>& candidates, double prevprevPosX, double prevprevPosY, double prevPosX, double prevPosY, double curPosX, double curPosY, double framerate, int prediction_range);

	int localPosX[4];
	int localPosY[4];
	int output_width_each_dir[4];

	int curPosX;
	int curPosY;
	int prevPosX;
	int prevPosY;
	int prevprevPosX;
	int prevprevPosY;
	int prevprevprevPosX;
	int prevprevprevPosY;

	int stride;
	std::vector<std::pair<int, int>> candidates_of_future_position;
	
	SliceSet slice_set[100][100];

	dim3 threadsPerBlock;
	dim3 blocksPerGrid_F;
	dim3 blocksPerGrid_R;
	dim3 blocksPerGrid_B;
	dim3 blocksPerGrid_L;
	cudaStream_t stream_main, stream_h2d;

	LRU_Cache* LRU;
	LFU_Window* window;
	
	MAIN_THREAD_STATE state_main_thread;
	H2D_THREAD_STATE state_h2d_thread;
	DISK_READ_THREAD_STATE state_disk_read_thread;

	std::mutex mtx;
	std::vector<std::thread> workers;
	bool use_window;

	LF_Config* config;
};

void cache_validity_check(int curPosX, int curPosY, SliceSet slice_set[][100], LFU_Window* window, LRU_Cache* LRU);
#endif