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
*	DoublyLinkedList를 위한 Node
*	Slice ID, 픽셀 데이터가 할당된 주소,
*	prev, next 포인터
*/

struct Slice
{
	SliceID id;
	uint8_t* data;
	Slice* prev;
	Slice* next;
};

/*	
*	Slice ID로 Slice 구조체를 찾기 위한 Hashmap과
*	Host memory에서 Slice의 픽셀 데이터가 Device에 복사된 주소를 관리하기 위한 h_devPtr_hashmap
*	Device가 해시맵에 접근 가능하도록 h_devPtr_hashmap으로부터 복사받는 d_devPtr_hashmap
*	'put' method는 Cache hit/miss 동작을 모두 처리
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