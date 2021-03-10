#include "LRU_Cache.h"

LRU_Cache::LRU_Cache(LF_Config* config, H2D_THREAD_STATE* h2d_thread_state)
{
	cudaError_t err;
	_config = config;
	state_h2d_thread = h2d_thread_state;

	head_odd = nullptr;
	tail_odd = nullptr;
	head_even = nullptr;
	tail_even = nullptr;

	interlaced_slice_size = _config->slice_size / 2;
	slice_hashmap_capacity = _config->LF_width / _config->slice_width * _config->LF_length * _config->num_LFs;
	current_LRU_size_odd = 0; // 현재 아이템 수
	current_LRU_size_even = 0; // 현재 아이템 수

	hashmap_odd = new Slice*[slice_hashmap_capacity];
	hashmap_even = new Slice*[slice_hashmap_capacity];
	err = cudaMallocHost((void**)&h_devPtr_hashmap_odd, slice_hashmap_capacity * sizeof(uint8_t*));
	err = cudaMallocHost((void**)&h_devPtr_hashmap_even, slice_hashmap_capacity * sizeof(uint8_t*));
	err = cudaMalloc((void**)&d_devPtr_hashmap_odd, slice_hashmap_capacity * sizeof(uint8_t*)); 
	err = cudaMalloc((void**)&d_devPtr_hashmap_even, slice_hashmap_capacity * sizeof(uint8_t*));
	assert(err == cudaSuccess);
	for (int i = 0; i < slice_hashmap_capacity; i++)
	{
		hashmap_odd[i] = nullptr;
		hashmap_even[i] = nullptr;
		h_devPtr_hashmap_odd[i] = nullptr;
		h_devPtr_hashmap_even[i] = nullptr;
	} // query를 host hashmap에 한 후, uint8_t* 결과만 d_ hashmap에 동기화
	slice_cache_capacity = (size_t)((double)get_devmem_freespace() * 0.8) / _config->slice_size;

	dmm_odd = new DeviceMemoryManager(slice_cache_capacity, interlaced_slice_size); // interlaced slice --> /=2
	dmm_even = new DeviceMemoryManager(slice_cache_capacity, interlaced_slice_size);

	printf("optimal slice width = %d\nslice_cache_capacity = %d\n", _config->slice_width, slice_cache_capacity);
}
LRU_Cache::~LRU_Cache()
{
	cudaError_t err;

	printf("Destruct LRU Cache\n");
	while (head_odd != tail_odd)
	{
		Slice* tmp = head_odd;
		h_devPtr_hashmap_odd[this->get_hashmap_location(tmp->id)] = nullptr;
		head_odd->next->prev = nullptr;
		head_odd = head_odd->next;
		delete tmp;
	}
	delete tail_odd;
	delete[] hashmap_odd;
	err = cudaFreeHost(h_devPtr_hashmap_odd);
	assert(err == cudaSuccess);
	err = cudaFree(d_devPtr_hashmap_odd);
	assert(err == cudaSuccess);

	while (head_even != tail_even)
	{
		Slice* tmp = head_even;
		h_devPtr_hashmap_even[this->get_hashmap_location(tmp->id)] = nullptr;
		head_even->next->prev = nullptr;
		head_even = head_even->next;
		delete tmp;
	}
	delete tail_even;
	delete[] hashmap_even;
	err = cudaFreeHost(h_devPtr_hashmap_even);
	assert(err == cudaSuccess);
	err = cudaFree(d_devPtr_hashmap_even);
	assert(err == cudaSuccess);
	
	delete dmm_odd;
	delete dmm_even;
}

size_t LRU_Cache::size(const INTERLACE_FIELD& field)
{
	if (field == ODD) return current_LRU_size_odd;
	else return current_LRU_size_even;
}

size_t LRU_Cache::get_slice_cache_capacity() {
	return this->slice_cache_capacity;
}

void LRU_Cache::enqueue_wait_slice(SliceID id, uint8_t* data, const INTERLACE_FIELD& field)
{
	if (field == ODD) {
		waiting_slice_odd.push(std::make_pair(id, data));
	}
	else {
		waiting_slice_even.push(std::make_pair(id, data));
	}
}

int LRU_Cache::put(const SliceID& id, uint8_t* data, const INTERLACE_FIELD& field)
{
	Slice** hashmap;
	uint8_t** h_devPtr_hashmap;
	uint8_t** d_devPtr_hashmap;
	Slice** head;
	Slice** tail;
	size_t* current_LRU_size;
	DeviceMemoryManager* dmm;

	if (field == ODD) {
		hashmap = hashmap_odd;
		h_devPtr_hashmap = h_devPtr_hashmap_odd;
		d_devPtr_hashmap = d_devPtr_hashmap_odd;
		head = &head_odd;
		tail = &tail_odd;
		current_LRU_size = &current_LRU_size_odd;
		dmm = dmm_odd;
	}
	else {
		hashmap = hashmap_even;
		h_devPtr_hashmap = h_devPtr_hashmap_even;
		d_devPtr_hashmap = d_devPtr_hashmap_even;
		head = &head_even;
		tail = &tail_even;
		current_LRU_size = &current_LRU_size_even;
		dmm = dmm_even;
	}

	int slice_location = query_hashmap(id, field);
	if (slice_location < 0) // Cache miss
	{
		int access_number = -1;
		if (*current_LRU_size >= slice_cache_capacity) {
			// cache full, evict head
			hashmap[get_hashmap_location((*head)->id)] = nullptr;
			h_devPtr_hashmap[get_hashmap_location((*head)->id)] = nullptr;
			access_number = (*head)->access_number;
			// dmm->return_access_number((*head)->access_number);
				
			Slice* tmp = (*head);
			(*head)->next->prev = nullptr;
			(*head) = (*head)->next;
			delete tmp;

			(*current_LRU_size)--;
		}

		Slice* slice = new Slice;
		slice->id = id;
		if (field == ODD) slice->odd_data = data;
		else slice->even_data = data;
		slice->access_number = access_number < 0 ? dmm->rent_access_number() : access_number;

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
		uint8_t* d_slice = dmm->get_empty_space(slice->access_number);
		
		cudaError_t err = cudaMemcpy(d_slice, data, this->interlaced_slice_size, cudaMemcpyHostToDevice); // data 복사
		assert(err == cudaSuccess);
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

void LRU_Cache::put(const SliceID& id, uint8_t* data, cudaStream_t stream, const INTERLACE_FIELD& field)
{
	Slice** hashmap;
	uint8_t** h_devPtr_hashmap;
	uint8_t** d_devPtr_hashmap;
	Slice** head;
	Slice** tail;
	size_t* current_LRU_size;
	DeviceMemoryManager* dmm;
	if (field == ODD) {
		hashmap = hashmap_odd;
		h_devPtr_hashmap = h_devPtr_hashmap_odd;
		d_devPtr_hashmap = d_devPtr_hashmap_odd;
		head = &head_odd;
		tail = &tail_odd;
		current_LRU_size = &current_LRU_size_odd;
		dmm = dmm_odd;
	}
	else {
		hashmap = hashmap_even;
		h_devPtr_hashmap = h_devPtr_hashmap_even;
		d_devPtr_hashmap = d_devPtr_hashmap_even;
		head = &head_even;
		tail = &tail_even;
		current_LRU_size = &current_LRU_size_even;
		dmm = dmm_even;
	}

	int slice_location = query_hashmap(id, field);
	if (slice_location < 0) {
		// cache miss
		int access_number = -1;
		if (*current_LRU_size >= slice_cache_capacity) {
			// cache full, evict head
			// printf("evict an item in cache\n");
			hashmap[get_hashmap_location((*head)->id)] = nullptr;
			h_devPtr_hashmap[get_hashmap_location((*head)->id)] = nullptr;
			access_number = (*head)->access_number;

			Slice* tmp = (*head);
			(*head)->next->prev = nullptr;
			(*head) = (*head)->next;
			delete tmp;

			(*current_LRU_size)--;
		}

		Slice* slice = new Slice;
		slice->id = id;
		if (field == ODD) slice->odd_data = data;
		else slice->even_data = data;
		slice->access_number = access_number < 0 ? dmm->rent_access_number() : access_number;

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
		uint8_t* d_slice = dmm->get_empty_space(slice->access_number);
		cudaError_t err = cudaMemcpyAsync(d_slice, data, this->interlaced_slice_size, cudaMemcpyHostToDevice, stream); // data 복사
		*state_h2d_thread = H2D_THREAD_RUNNING;
		err = cudaStreamSynchronize(stream); // this stream must block the host code
		assert(err == cudaSuccess);

		h_devPtr_hashmap[get_hashmap_location(slice->id)] = d_slice; // hashmap에 저장된 주소공간을 할당 후

		(*current_LRU_size)++;
		*state_h2d_thread = H2D_THREAD_WAIT;
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

int LRU_Cache::synchronize_HashmapOfPtr(LFU_Window& window, cudaStream_t stream)
{
	while (!waiting_slice_odd.empty())
	{
		SliceID id = waiting_slice_odd.front().first;
		uint8_t* data = waiting_slice_odd.front().second;
		put(id, data, ODD);
		waiting_slice_odd.pop();
	}
	cudaError_t err = cudaMemcpyAsync(d_devPtr_hashmap_odd, h_devPtr_hashmap_odd, _config->LF_width / _config->slice_width * _config->LF_length * _config->num_LFs * sizeof(uint8_t*), cudaMemcpyHostToDevice, stream);
	assert(err == cudaSuccess);
	cudaStreamSynchronize(stream);

	if (window.pinned_memory_status == PINNED_LFU_EVEN_AVAILABLE) {
		while (!waiting_slice_even.empty())
		{
			SliceID id = waiting_slice_even.front().first;
			uint8_t* data = waiting_slice_even.front().second;
			put(id, data, EVEN);
			waiting_slice_even.pop();
		}
		cudaError_t err = cudaMemcpyAsync(d_devPtr_hashmap_even, h_devPtr_hashmap_even, _config->LF_width / _config->slice_width * _config->LF_length * _config->num_LFs * sizeof(uint8_t*), cudaMemcpyHostToDevice, stream);
		assert(err == cudaSuccess);
		cudaStreamSynchronize(stream);

		return 1;
	}
	else return 0;
}

int LRU_Cache::get_hashmap_location(const SliceID& id)
{
	return id.lf_number * (_config->LF_width / _config->slice_width) * _config->LF_length + id.image_number * (_config->LF_width / _config->slice_width) + id.slice_number;
}

int LRU_Cache::query_hashmap(const SliceID& id, const INTERLACE_FIELD& field)
{
	Slice** hashmap;
	if (field == ODD) hashmap = hashmap_odd;
	else hashmap = hashmap_even;
	int slice_location = get_hashmap_location(id);

	if (hashmap[slice_location] == nullptr) return -1;
	else return slice_location;
}

bool LRU_Cache::isFull(const INTERLACE_FIELD& field)
{
	if (field == ODD) {
		if (this->current_LRU_size_odd < slice_cache_capacity)
			return false;
		else return true;
	}
	else {
		if (this->current_LRU_size_even < slice_cache_capacity)
			return false;
		else return true;
	}
}

uint8_t* LRU_Cache::find_slice_in_hashmap(SliceID id)
{
	if (h_devPtr_hashmap_odd[get_hashmap_location(id)] == nullptr) return nullptr;
	else return h_devPtr_hashmap_odd[get_hashmap_location(id)];
}

int LRU_Cache::flush_cache() // 미완성
{
	while (head_odd != tail_odd)
	{
		Slice* tmp = head_odd;
		this->hashmap_odd[this->get_hashmap_location(tmp->id)] = nullptr;
		h_devPtr_hashmap_odd[this->get_hashmap_location(tmp->id)] = nullptr;
		head_odd->next->prev = nullptr;
		head_odd = head_odd->next;
		delete tmp;
	}
	delete tail_odd;
	while (head_even != tail_even)
	{
		Slice* tmp = head_even;
		this->hashmap_even[this->get_hashmap_location(tmp->id)] = nullptr;
		h_devPtr_hashmap_even[this->get_hashmap_location(tmp->id)] = nullptr;
		head_even->next->prev = nullptr;
		head_even = head_even->next;
		delete tmp;
	}
	delete tail_even;

	this->current_LRU_size_odd = 0;
	this->current_LRU_size_even = 0;
	this->dmm_odd->init();
	this->dmm_even->init();

	return 0;
}
