#ifndef LF_UTILS_CUH_
#define LF_UTILS_CUH_ 
// Properties->CUDA C/C++->Common->generate relocatable device code=Yes

#include "LFUtils.h"
#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"

struct SliceID {
	int lf_number;
	int image_number;
	int slice_number;
};

/*	
*	DoublyLinkedList�� ���� Node
*	Slice ID, �ȼ� �����Ͱ� �Ҵ�� �ּ�,
*	prev, next ������
*/

struct Slice
{
	SliceID id;
	uint8_t* data;
	Slice* prev;
	Slice* next;
};

/*	
*	Slice ID�� Slice ����ü�� ã�� ���� Hashmap��
*	Host memory���� Slice�� �ȼ� �����Ͱ� Device�� ����� �ּҸ� �����ϱ� ���� h_devPtr_hashmap
*	Device�� �ؽøʿ� ���� �����ϵ��� h_devPtr_hashmap���κ��� ����޴� d_devPtr_hashmap
*	'put' method�� Cache hit/miss ������ ��� ó��
*/

class LRUCache {
public:
	LRUCache(int num_limit_HashingLF, int num_limit_slice);
	~LRUCache();

	int query_hashmap(const SliceID& id);
	void enqueue_wait_slice(SliceID id, uint8_t* data);

	int put(const SliceID& id, uint8_t* data);
	void put(const SliceID& id, uint8_t* data, cudaStream_t stream, H2D_THREAD_STATE& h2d_thread_state); // for Worker thread

	int synchronize_HashmapOfPtr(std::vector<Interlaced_LF>& window, cudaStream_t stream);
	int size();
	Slice** hashmap;
	uint8_t** h_devPtr_hashmap;
	uint8_t** d_devPtr_hashmap;
private:
	int get_hashmap_location(const SliceID& id);

	Slice* head;
	Slice* tail;

	int current_LRU_size;
	int num_limit_slice;
	int num_limit_HashingLF;

	std::queue <std::pair<SliceID, uint8_t*>> waiting_slice;
};

__device__ int dev_SignBitMasking(int l, int r);

__device__ int dev_Clamp(int val, int min, int max);

__device__ float dev_rad2deg(float rad);

__device__ float dev_deg2rad(float deg);

#endif