#ifndef LRU_CACHE_H_
#define LRU_CACHE_H_

#include "LF_Utils.cuh"
#include "LFU_Window.h"

struct SliceID {
	FOUR_DIRECTION direction;
	int image_number;
	int slice_number;
};

struct Slice
{
	SliceID id;
	// INTERLACE_FIELD field;
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

	int synchronize_HashmapOfPtr(LFU_Window& window, cudaStream_t stream, const READ_DISK_THREAD_STATE& read_disk_thread_state);
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
#endif