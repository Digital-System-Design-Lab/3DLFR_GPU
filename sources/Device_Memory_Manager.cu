#include "Device_Memory_Manager.cuh"

DeviceMemoryManager::DeviceMemoryManager(const size_t& num_of_slice, const size_t& slice_size)
{
	cudaError_t err;
	err = cudaMalloc((void**)&dev_slice_buffer, slice_size * num_of_slice);
	assert(err == cudaSuccess);
	access_number_set.resize(num_of_slice);
	this->num_of_slice = num_of_slice;
	this->slice_size = slice_size;
	this->last_access_number = 0;
}

DeviceMemoryManager::~DeviceMemoryManager()
{
	cudaError_t err;
	err = cudaFree(dev_slice_buffer);
	assert(err == cudaSuccess);
}

int DeviceMemoryManager::rent_access_number()
{
	for (int i = last_access_number; i < num_of_slice; i++) {
		if (access_number_set[i] < 1) {
			access_number_set[i] = 1;
			last_access_number = i;

			return i;
		}
	}
}
void DeviceMemoryManager::return_access_number(const size_t& num)
{
	access_number_set[num] = 0;
}

uint8_t* DeviceMemoryManager::get_empty_space(const size_t& access_number)
{
	return dev_slice_buffer + access_number * slice_size;
}

void DeviceMemoryManager::init()
{
	cudaMemset(this->dev_slice_buffer, 0, slice_size * num_of_slice);
	this->last_access_number = 0;
}