#include "LFUtils.cuh"

__device__ int dev_SignBitMasking(int l, int r)
{
	return !!((l - r) & 0x80000000); // if l < r : return 1
}

__device__ int dev_Clamp(int val, int min, int max)
{
	return	!dev_SignBitMasking(val, max) * max +
		dev_SignBitMasking(val, max) * (dev_SignBitMasking(val, min) * min + !dev_SignBitMasking(val, min) * val);
}

__device__ float dev_rad2deg(float rad)
{
	return (rad * 180.0f / 3.14159274f);
}

__device__ float dev_deg2rad(float deg)
{
	return (deg * 3.14159274f / 180.0f);
}

LRUCache::LRUCache(int num_limit_HashingLF, int num_limit_slice)
{
	this->head = nullptr;
	this->tail = nullptr;

	this->num_limit_HashingLF = num_limit_HashingLF; // 해싱 가능한 LF의 범위 (LF 0부터 시작)
	this->num_limit_slice = num_limit_slice;
	this->current_LRU_size = 0; // 현재 아이템 수

	hashmap = new Slice*[g_width / g_slice_width * g_length * num_limit_HashingLF];
	for (int i = 0; i < g_width / g_slice_width * g_length * num_limit_HashingLF; i++)
	{
		hashmap[i] = nullptr;
	} // query를 host hashmap에 한 후, uint8_t* 결과만 d_ hashmap에 동기화

	cudaMallocHost((void**)&h_devPtr_hashmap, g_width / g_slice_width * g_length * num_limit_HashingLF * sizeof(uint8_t*));
	for (int i = 0; i < g_width / g_slice_width * g_length * num_limit_HashingLF; i++)
	{
		h_devPtr_hashmap[i] = nullptr;
	} // query를 host hashmap에 한 후, uint8_t* 결과만 d_ hashmap에 동기화

	cudaMalloc((void**)&d_devPtr_hashmap, g_width / g_slice_width * g_length * num_limit_HashingLF * sizeof(uint8_t*));
}
LRUCache::~LRUCache()
{
	while (head != tail)
	{
		Slice* tmp = head;

		uint8_t* d_data = h_devPtr_hashmap[this->get_hashmap_location(tmp->id)];
		h_devPtr_hashmap[this->get_hashmap_location(tmp->id)] = nullptr;
		cudaFree(d_data);

		head->next->prev = nullptr;
		head = head->next;
		delete tmp;
	}
	delete tail;
	delete[] hashmap;
	cudaFreeHost(h_devPtr_hashmap);
	cudaFree(d_devPtr_hashmap);
}

int LRUCache::size()
{
	return this->current_LRU_size;
}

void LRUCache::enqueue_wait_slice(SliceID id, uint8_t* data)
{
	waiting_slice.push(std::make_pair(id, data));
}

int LRUCache::put(const SliceID& id, uint8_t* data)
{
	int slice_location = query_hashmap(id);
	if (slice_location < 0) // Cache miss
	{
		if (current_LRU_size >= num_limit_slice) {
			// cache full, evict head
			hashmap[get_hashmap_location(head->id)] = nullptr;

			Slice* tmp = head;
			head->next->prev = nullptr;
			head = head->next;
			delete tmp;

			uint8_t* d_tmp = h_devPtr_hashmap[get_hashmap_location(head->id)];
			h_devPtr_hashmap[get_hashmap_location(head->id)] = nullptr;
			cudaFree(d_tmp);

			current_LRU_size--;
		}

		Slice* slice = new Slice;
		slice->id = id;
		slice->data = data;

		if (head == nullptr) {
			// No item in cache
			head = slice;
			tail = slice;
			head->prev = nullptr;
			tail->next = nullptr;
		}
		else {
			tail->next = slice;
			slice->prev = tail;
			tail = slice;
			tail->next = nullptr;
		}
		hashmap[get_hashmap_location(slice->id)] = slice;
		uint8_t* d_slice;
		cudaMalloc((void**)&d_slice, g_slice_size);
		cudaMemcpy(d_slice, slice->data, g_slice_size, cudaMemcpyHostToDevice); // data 복사
		h_devPtr_hashmap[get_hashmap_location(slice->id)] = d_slice; // hashmap에 저장된 주소공간을 할당 후
		
		current_LRU_size++;

		return 0;
	}
	else {
		// Cache hit 
		Slice* slice = hashmap[slice_location];
		if (slice != tail) {
			if (slice == head) {
				head = head->next;
				head->prev = nullptr;
			}
			if (slice->prev != nullptr) slice->prev->next = slice->next;
			if (slice->next != nullptr) slice->next->prev = slice->prev;
			slice->prev = tail;
			tail->next = slice;
			slice->next = nullptr;
			tail = slice;
		}

		return 1;
	}
}

void LRUCache::put(const SliceID& id, uint8_t* data, cudaStream_t stream, H2D_THREAD_STATE& h2d_thread_state)
{
	int slice_location = query_hashmap(id);
	if (slice_location < 0) {
		// cache miss
		if (current_LRU_size >= num_limit_slice) {
			// cache full, evict head
			hashmap[get_hashmap_location(head->id)] = nullptr;

			Slice* tmp = head;
			head->next->prev = nullptr;
			head = head->next;
			delete tmp;

			uint8_t* d_tmp = h_devPtr_hashmap[get_hashmap_location(head->id)];
			h_devPtr_hashmap[get_hashmap_location(head->id)] = nullptr;
			cudaFree(d_tmp);

			current_LRU_size--;
		}

		Slice* slice = new Slice;
		slice->id = id;
		slice->data = data;

		if (head == nullptr) {
			// No item in cache
			head = slice;
			tail = slice;
			head->prev = nullptr;
			tail->next = nullptr;
		}
		else {
			tail->next = slice;
			slice->prev = tail;
			tail = slice;
			tail->next = nullptr;
		}

		hashmap[get_hashmap_location(slice->id)] = slice;
		uint8_t* d_slice; 
		cudaError_t err = cudaMalloc((void**)&d_slice, g_slice_size); // slice를 위한 device memory 할당
		err = cudaMemcpyAsync(d_slice, slice->data, g_slice_size, cudaMemcpyHostToDevice, stream); // data 복사
		h2d_thread_state = H2D_THREAD_RUNNING;
		cudaStreamSynchronize(stream); // this stream must block the host code
		h_devPtr_hashmap[get_hashmap_location(slice->id)] = d_slice; // hashmap에 저장된 주소공간을 할당 후

		current_LRU_size++;
		h2d_thread_state = H2D_THREAD_WAIT;
	}
	else {
		// Cache hit 
		Slice* slice = hashmap[slice_location];
		if (slice != tail) {
			if (slice == head) {
				head = head->next;
				head->prev = nullptr;
			}
			if (slice->prev != nullptr) slice->prev->next = slice->next;
			if (slice->next != nullptr) slice->next->prev = slice->prev;
			slice->prev = tail;
			tail->next = slice;
			slice->next = nullptr;
			tail = slice;
		}
	}
}

int LRUCache::synchronize_HashmapOfPtr(std::vector<Interlaced_LF>& window, cudaStream_t stream)
{
	while (!waiting_slice.empty())
	{
		SliceID id = waiting_slice.front().first;
		uint8_t* data = waiting_slice.front().second;
		Interlaced_LF* LF = get_LF_from_Window(window, id.lf_number);

		if (LF->progress == LF_READ_PROGRESS_PREPARED) {
			put(id, data);
			waiting_slice.pop();
		}
	}
	cudaError_t err = cudaMemcpyAsync(d_devPtr_hashmap, h_devPtr_hashmap, g_width / g_slice_width * g_length * num_limit_HashingLF * sizeof(uint8_t*), cudaMemcpyHostToDevice, stream);
	assert(err == cudaSuccess);
	cudaStreamSynchronize(stream);

	return 0;
}

int LRUCache::get_hashmap_location(const SliceID& id)
{
	return id.lf_number * (g_width / g_slice_width) * g_length + id.image_number * (g_width / g_slice_width) + id.slice_number;
}

int LRUCache::query_hashmap(const SliceID& id)
{
	int slice_location = get_hashmap_location(id);

	if (hashmap[slice_location] == nullptr) return -1;
	else return slice_location;
}
