#ifndef LRU_CACHE_H_
#define LRU_CACHE_H_

#include "LF_Utils.cuh"
#include "LFU_Window.h"
#include "Device_Memory_Manager.cuh"

struct SliceID {
	int lf_number;
	int image_number;
	int slice_number;
};

struct Slice
{
	SliceID id;
	size_t access_number;
	// INTERLACE_FIELD field;
	uint8_t* odd_data;
	uint8_t* even_data;
	Slice* prev;
	Slice* next;
};

class LRU_Cache {
public:
	LRU_Cache(LF_Config* config, H2D_THREAD_STATE* h2d_thread_state);
	~LRU_Cache();

	int query_hashmap(const SliceID& id, const INTERLACE_FIELD& field);
	void enqueue_wait_slice(SliceID id, uint8_t* data, const INTERLACE_FIELD& field);

	int put(const SliceID& id, uint8_t* data, const INTERLACE_FIELD& field);
	void put(const SliceID& id, uint8_t* data, cudaStream_t stream, const INTERLACE_FIELD& field); // for Worker thread

	int synchronize_HashmapOfPtr(LFU_Window& window, cudaStream_t stream);
	int size(const INTERLACE_FIELD& field);

	bool isFull(const INTERLACE_FIELD& field);

	uint8_t* find_slice_in_hashmap(SliceID id);

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

	size_t current_LRU_size_odd;
	size_t current_LRU_size_even;
	size_t slice_cache_capacity;
	size_t slice_hashmap_capacity;
	size_t interlaced_slice_size;

	std::queue <std::pair<SliceID, uint8_t*>> waiting_slice_odd;
	std::queue <std::pair<SliceID, uint8_t*>> waiting_slice_even;

	DeviceMemoryManager* dmm_odd;
	DeviceMemoryManager* dmm_even;

	LF_Config* _config;

	H2D_THREAD_STATE* state_h2d_thread;
};
#endif