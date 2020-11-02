#ifndef LF_UTILS_CUH_
#define LF_UTILS_CUH_ 
// Properties->CUDA C/C++->Common->generate relocatable device code=Yes

#include "LFUtils.h"
#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"

enum INTERLACE_FIELD {
	ODD = 0,
	EVEN = 1
};

struct SliceID {
	int lf_number;
	int image_number;
	int slice_number;
};

struct Slice
{
	SliceID id;
	// uint8_t* data;
	uint8_t* odd_data;
	uint8_t* even_data;
	Slice* prev;
	Slice* next;
};

class LRUCache {
public:
	LRUCache(int num_limit_HashingLF, int num_limit_slice);
	~LRUCache();

	int query_hashmap(const SliceID& id, const INTERLACE_FIELD& field);
	void enqueue_wait_slice(SliceID id, uint8_t* data, const INTERLACE_FIELD& field);

	int put(const SliceID& id, uint8_t* data, const INTERLACE_FIELD& field);
	void put(const SliceID& id, uint8_t* data, cudaStream_t stream, H2D_THREAD_STATE& p_h2d_thread_state, const INTERLACE_FIELD& field); // for Worker thread

	int synchronize_HashmapOfPtr(std::vector<Interlaced_LF>& window, cudaStream_t stream, const READ_DISK_THREAD_STATE& read_disk_thread_state);
	int size(const INTERLACE_FIELD& field);

	Slice** hashmap_odd;
	uint8_t** h_devPtr_hashmap_odd;
	uint8_t** d_devPtr_hashmap_odd;

	Slice** hashmap_even;
	uint8_t** h_devPtr_hashmap_even;
	uint8_t** d_devPtr_hashmap_even;
private:
	int get_hashmap_location(const SliceID& id);

	Slice* head_odd;
	Slice* tail_odd;
	Slice* head_even;
	Slice* tail_even;

	int current_LRU_size_odd;
	int current_LRU_size_even;
	int num_limit_slice;
	int num_limit_HashingLF;

	std::queue <std::pair<SliceID, uint8_t*>> waiting_slice_odd;
	std::queue <std::pair<SliceID, uint8_t*>> waiting_slice_even;
};

__device__ int dev_SignBitMasking(int l, int r);

__device__ int dev_Clamp(int val, int min, int max);

__device__ float dev_rad2deg(float rad);

__device__ float dev_deg2rad(float deg);

#endif