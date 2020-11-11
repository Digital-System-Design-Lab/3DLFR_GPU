#include "Device_Memory_Manager.cuh"

DeviceMemoryManager::DeviceMemoryManager(const size_t& num_of_slice, const size_t& slice_size)
{
	cudaMalloc((void**)&dev_slice_buffer, slice_size * num_of_slice);
	for (size_t i = 0; i < num_of_slice; i++)
		access_number_set.push_back(0);
	this->num_of_slice = num_of_slice;
}

DeviceMemoryManager::~DeviceMemoryManager()
{
	cudaFree(dev_slice_buffer);
}

int DeviceMemoryManager::rent_access_number()
{
	for (int i = 0; i < num_of_slice; i++) {
		if (access_number_set[i] < 1) {
			access_number_set[i] = 1;
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
	return dev_slice_buffer + access_number * g_slice_size;
}