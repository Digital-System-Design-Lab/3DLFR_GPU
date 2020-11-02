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
	this->head_odd = nullptr;
	this->tail_odd = nullptr;
	this->head_even = nullptr;
	this->tail_even = nullptr;

	this->num_limit_HashingLF = num_limit_HashingLF; // 해싱 가능한 LF의 범위 (LF 0부터 시작)
	this->num_limit_slice = num_limit_slice;
	this->current_LRU_size_odd = 0; // 현재 아이템 수
	this->current_LRU_size_even = 0; // 현재 아이템 수

	hashmap_odd = new Slice*[g_width / g_slice_width * g_length * num_limit_HashingLF];
	for (int i = 0; i < g_width / g_slice_width * g_length * num_limit_HashingLF; i++)
	{
		hashmap_odd[i] = nullptr;
	} // query를 host hashmap에 한 후, uint8_t* 결과만 d_ hashmap에 동기화

	cudaMallocHost((void**)&h_devPtr_hashmap_odd, g_width / g_slice_width * g_length * num_limit_HashingLF * sizeof(uint8_t*));
	for (int i = 0; i < g_width / g_slice_width * g_length * num_limit_HashingLF; i++)
	{
		h_devPtr_hashmap_odd[i] = nullptr;
	} // query를 host hashmap에 한 후, uint8_t* 결과만 d_ hashmap에 동기화

	cudaMalloc((void**)&d_devPtr_hashmap_odd, g_width / g_slice_width * g_length * num_limit_HashingLF * sizeof(uint8_t*));

	hashmap_even = new Slice*[g_width / g_slice_width * g_length * num_limit_HashingLF];
	for (int i = 0; i < g_width / g_slice_width * g_length * num_limit_HashingLF; i++)
	{
		hashmap_even[i] = nullptr;
	} // query를 host hashmap에 한 후, uint8_t* 결과만 d_ hashmap에 동기화

	cudaMallocHost((void**)&h_devPtr_hashmap_even, g_width / g_slice_width * g_length * num_limit_HashingLF * sizeof(uint8_t*));
	for (int i = 0; i < g_width / g_slice_width * g_length * num_limit_HashingLF; i++)
	{
		h_devPtr_hashmap_even[i] = nullptr;
	} // query를 host hashmap에 한 후, uint8_t* 결과만 d_ hashmap에 동기화

	cudaMalloc((void**)&d_devPtr_hashmap_even, g_width / g_slice_width * g_length * num_limit_HashingLF * sizeof(uint8_t*));
}
LRUCache::~LRUCache()
{
	while (head_odd != tail_odd)
	{
		Slice* tmp = head_odd;

		uint8_t* d_data = h_devPtr_hashmap_odd[this->get_hashmap_location(tmp->id)];
		h_devPtr_hashmap_odd[this->get_hashmap_location(tmp->id)] = nullptr;
		cudaFree(d_data);

		head_odd->next->prev = nullptr;
		head_odd = head_odd->next;
		delete tmp;
	}
	delete tail_odd;
	delete[] hashmap_odd;
	cudaFreeHost(h_devPtr_hashmap_odd);
	cudaFree(d_devPtr_hashmap_odd);

	while (head_even != tail_even)
	{
		Slice* tmp = head_even;

		uint8_t* d_data = h_devPtr_hashmap_even[this->get_hashmap_location(tmp->id)];
		h_devPtr_hashmap_even[this->get_hashmap_location(tmp->id)] = nullptr;
		cudaFree(d_data);

		head_even->next->prev = nullptr;
		head_even = head_even->next;
		delete tmp;
	}
	delete tail_even;
	delete[] hashmap_even;
	cudaFreeHost(h_devPtr_hashmap_even);
	cudaFree(d_devPtr_hashmap_even);
}

int LRUCache::size(const INTERLACE_FIELD& field)
{
	if (field == ODD) return current_LRU_size_odd;
	else return current_LRU_size_even;
}

void LRUCache::enqueue_wait_slice(SliceID id, uint8_t* data, const INTERLACE_FIELD& field)
{
	if (field == ODD) {
		waiting_slice_odd.push(std::make_pair(id, data));
	}
	else {
		waiting_slice_even.push(std::make_pair(id, data));
	}
}

int LRUCache::put(const SliceID& id, uint8_t* data, const INTERLACE_FIELD& field)
{
	Slice** hashmap;
	uint8_t** h_devPtr_hashmap;
	uint8_t** d_devPtr_hashmap;
	Slice** head;
	Slice** tail;
	int* current_LRU_size;
	if (field == ODD) {
		hashmap = hashmap_odd;
		h_devPtr_hashmap = h_devPtr_hashmap_odd;
		d_devPtr_hashmap = d_devPtr_hashmap_odd;
		head = &head_odd;
		tail = &tail_odd;
		current_LRU_size = &current_LRU_size_odd;
	}
	else {
		hashmap = hashmap_even;
		h_devPtr_hashmap = h_devPtr_hashmap_even;
		d_devPtr_hashmap = d_devPtr_hashmap_even;
		head = &head_even;
		tail = &tail_even;
		current_LRU_size = &current_LRU_size_even;
	}

	int slice_location = query_hashmap(id, field);
	if (slice_location < 0) // Cache miss
	{
		if (*current_LRU_size >= num_limit_slice) {
			// cache full, evict head
			hashmap[get_hashmap_location((*head)->id)] = nullptr;

			Slice* tmp = (*head);
			(*head)->next->prev = nullptr;
			(*head) = (*head)->next;
			delete tmp;

			uint8_t* d_tmp = h_devPtr_hashmap[get_hashmap_location((*head)->id)];
			h_devPtr_hashmap[get_hashmap_location((*head)->id)] = nullptr;
			cudaFree(d_tmp);

			(*current_LRU_size)--;
		}

		Slice* slice = new Slice;
		slice->id = id;
		if (field == ODD) slice->odd_data = data;
		else slice->even_data = data;

		if ((*head) == nullptr) {
			// No item in cache
			(*head) = slice;
			(*tail) = slice;
			(*head)->prev = nullptr;
			(*tail)->next = nullptr;
		}
		else {
			(*tail)->next = slice;
			slice->prev = (*tail);
			(*tail) = slice;
			(*tail)->next = nullptr;
		}
		hashmap[get_hashmap_location(slice->id)] = slice;
		uint8_t* d_slice;
		cudaMalloc((void**)&d_slice, g_slice_size / 2);
		cudaMemcpy(d_slice, data, g_slice_size / 2, cudaMemcpyHostToDevice); // data 복사
		h_devPtr_hashmap[get_hashmap_location(slice->id)] = d_slice; // hashmap에 저장된 주소공간을 할당 후

		(*current_LRU_size)++;

		return 0;
	}
	else {
		// Cache hit 
		Slice* slice = hashmap[slice_location];
		if (slice != (*tail)) {
			if (slice == (*head)) {
				(*head) = (*head)->next;
				(*head)->prev = nullptr;
			}
			if (slice->prev != nullptr) slice->prev->next = slice->next;
			if (slice->next != nullptr) slice->next->prev = slice->prev;
			slice->prev = (*tail);
			(*tail)->next = slice;
			slice->next = nullptr;
			(*tail) = slice;
		}

		return 1;
	}
}

void LRUCache::put(const SliceID& id, uint8_t* data, cudaStream_t stream, H2D_THREAD_STATE& p_h2d_thread_state, const INTERLACE_FIELD& field)
{
	Slice** hashmap;
	uint8_t** h_devPtr_hashmap;
	uint8_t** d_devPtr_hashmap;
	Slice** head;
	Slice** tail;
	int* current_LRU_size;

	if (field == ODD) {
		hashmap = hashmap_odd;
		h_devPtr_hashmap = h_devPtr_hashmap_odd;
		d_devPtr_hashmap = d_devPtr_hashmap_odd;
		head = &head_odd;
		tail = &tail_odd;
		current_LRU_size = &current_LRU_size_odd;
	}
	else {
		hashmap = hashmap_even;
		h_devPtr_hashmap = h_devPtr_hashmap_even;
		d_devPtr_hashmap = d_devPtr_hashmap_even;
		head = &head_even;
		tail = &tail_even;
		current_LRU_size = &current_LRU_size_even;
	}

	int slice_location = query_hashmap(id, field);
	if (slice_location < 0) {
		// cache miss
		if (*current_LRU_size >= num_limit_slice) {
			// cache full, evict head
			hashmap[get_hashmap_location((*head)->id)] = nullptr;

			Slice* tmp = (*head);
			(*head)->next->prev = nullptr;
			(*head) = (*head)->next;
			delete tmp;

			uint8_t* d_tmp = h_devPtr_hashmap[get_hashmap_location((*head)->id)];
			h_devPtr_hashmap[get_hashmap_location((*head)->id)] = nullptr;
			cudaFree(d_tmp);

			(*current_LRU_size)--;
		}

		Slice* slice = new Slice;
		slice->id = id;
		if (field) slice->odd_data = data;
		else slice->even_data = data;

		if ((*head) == nullptr) {
			// No item in cache
			(*head) = slice;
			(*tail) = slice;
			(*head)->prev = nullptr;
			(*tail)->next = nullptr;
		}
		else {
			(*tail)->next = slice;
			slice->prev = (*tail);
			(*tail) = slice;
			(*tail)->next = nullptr;
		}

		hashmap[get_hashmap_location(slice->id)] = slice;
		uint8_t* d_slice;
		cudaError_t err = cudaMalloc((void**)&d_slice, g_slice_size); // slice를 위한 device memory 할당
		err = cudaMemcpyAsync(d_slice, data, g_slice_size, cudaMemcpyHostToDevice, stream); // data 복사
		p_h2d_thread_state = H2D_THREAD_RUNNING;
		cudaStreamSynchronize(stream); // this stream must block the host code
		h_devPtr_hashmap[get_hashmap_location(slice->id)] = d_slice; // hashmap에 저장된 주소공간을 할당 후

		(*current_LRU_size)++;
		p_h2d_thread_state = H2D_THREAD_WAIT;
	}
	else {
		// Cache hit 
		Slice* slice = hashmap[slice_location];
		if (slice != (*tail)) {
			if (slice == (*head)) {
				(*head) = (*head)->next;
				(*head)->prev = nullptr;
			}
			if (slice->prev != nullptr) slice->prev->next = slice->next;
			if (slice->next != nullptr) slice->next->prev = slice->prev;
			slice->prev = (*tail);
			(*tail)->next = slice;
			slice->next = nullptr;
			(*tail) = slice;
		}
	}
}

int LRUCache::synchronize_HashmapOfPtr(std::vector<Interlaced_LF>& window, cudaStream_t stream, const READ_DISK_THREAD_STATE& read_disk_thread_state)
{
	while (!waiting_slice_odd.empty())
	{
		SliceID id = waiting_slice_odd.front().first;
		uint8_t* data = waiting_slice_odd.front().second;
		Interlaced_LF* LF = get_LF_from_Window(window, id.lf_number);

		if (LF->progress >= LF_READ_PROGRESS_ODD_FIELD_PREPARED) {
			put(id, data, ODD);
			waiting_slice_odd.pop();
		}
	}
	cudaError_t err = cudaMemcpyAsync(d_devPtr_hashmap_odd, h_devPtr_hashmap_odd, g_width / g_slice_width * g_length * num_limit_HashingLF * sizeof(uint8_t*), cudaMemcpyHostToDevice, stream);
	assert(err == cudaSuccess);
	cudaStreamSynchronize(stream);

	bool isEvenAvailable = true;

	if (read_disk_thread_state >= READ_DISK_THREAD_NEIGHBOR_LF_READING)
	{
		while (!waiting_slice_even.empty())
		{
			SliceID id = waiting_slice_even.front().first;
			uint8_t* data = waiting_slice_even.front().second;
			Interlaced_LF* LF = get_LF_from_Window(window, id.lf_number);

			if (LF->progress == LF_READ_PROGRESS_EVEN_FIELD_PREPARED) {
				put(id, data, EVEN);
				waiting_slice_even.pop();
				isEvenAvailable = true;
			}
			else {
				isEvenAvailable = false;
				break;
			}
		}
		if (isEvenAvailable) {
			cudaError_t err = cudaMemcpyAsync(d_devPtr_hashmap_even, h_devPtr_hashmap_even, g_width / g_slice_width * g_length * num_limit_HashingLF * sizeof(uint8_t*), cudaMemcpyHostToDevice, stream);
			assert(err == cudaSuccess);
			cudaStreamSynchronize(stream);

			return 1;
		}
	}

	return 0;
}

int LRUCache::get_hashmap_location(const SliceID& id)
{
	return id.lf_number * (g_width / g_slice_width) * g_length + id.image_number * (g_width / g_slice_width) + id.slice_number;
}

int LRUCache::query_hashmap(const SliceID& id, const INTERLACE_FIELD& field)
{
	Slice** hashmap;
	if (field == ODD) hashmap = hashmap_odd;
	else hashmap = hashmap_even;
	int slice_location = get_hashmap_location(id);

	if (hashmap[slice_location] == nullptr) return -1;
	else return slice_location;
}