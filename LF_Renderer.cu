#include "LF_Renderer.cuh"

LF_Renderer::LF_Renderer(const std::string& pixel_range_path, const std::string& LF_path, const size_t& limit_cache_size, const size_t& limit_LF, const int& initPosX, const int& initPosY)
{
	const size_t light_field_size = 4096 * 2048 * 3 * 50 / 2;

	curPosX = initPosX;
	curPosY = initPosY;
	prevPosX = curPosX;
	prevPosY = curPosY;
	this->nbrPosition.resize(8);
	getNeighborList(nbrPosition, curPosX, curPosY);

	load_slice_set(this->slice_set, pixel_range_path);
	
	LRUCache* lru_cache = new LRUCache(limit_LF, limit_cache_size, &io_config);
	this->LRU = lru_cache;
	LFU_Window* lfu_window = new LFU_Window(curPosX, curPosY, light_field_size, LF_path);
	this->window = lfu_window;
	
	this->u_synthesized_view = alloc_uint8(this->io_config.output_width * this->io_config.LF_height * 3, "unified");

	state_main_thread = MAIN_THREAD_INIT;
	state_h2d_thread = H2D_THREAD_INIT;

	cudaStreamCreate(&stream_main);
	cudaStreamCreate(&stream_h2d);

	workers.push_back(std::thread(&LF_Renderer::loop_nbrs_h2d, this, std::ref(*LRU), std::ref(*window), slice_set, std::ref(nbrPosition), stream_h2d, std::ref(state_h2d_thread), std::ref(state_main_thread), std::ref(mtx)));
	workers.push_back(std::thread(&LF_Renderer::loop_read_disk, this, std::ref(*window), std::ref(prevPosX), std::ref(prevPosY), std::ref(curPosX), std::ref(curPosY), std::ref(light_field_size), std::ref(state_main_thread)));
}

LF_Renderer::~LF_Renderer() {
	printf("Destruct LF Renderer\n");
	for(std::vector<std::thread>::iterator it = workers.begin(); it != workers.end(); it++) { 
		it->join();
	}
	delete this->LRU;
	delete this->window;
	
	free_uint8(u_synthesized_view, "unified");
	cudaStreamDestroy(stream_main);
	cudaStreamDestroy(stream_h2d);
}

uint8_t* LF_Renderer::do_rendering(const int& newPosX, const int& newPosY) {
	this->prevPosX = curPosX;
	this->prevPosY = curPosY;
	this->curPosX = newPosX;
	this->curPosY = newPosY;
	curPosX = clamp(curPosX, 101, 499);
	curPosY = clamp(curPosY, 101, 5499);

	set_rendering_range(localPosX, localPosY, output_width_each_dir, curPosX, curPosY);

	state_main_thread = MAIN_THREAD_WAIT;
	getNeighborList(nbrPosition, curPosX, curPosY);

	state_main_thread = MAIN_THREAD_H2D;

	mtx.lock();
	int ret = cache_slice(*LRU, *window, slice_set, curPosX, curPosY);
	int mode = LRU->synchronize_HashmapOfPtr(*window, stream_main);
	mtx.unlock();

	state_main_thread = MAIN_THREAD_RENDERING;
	int twid = 2;
	int thei = 32;
	dim3 threadsPerBlock(twid, thei);
	// interlace mode -> block shape : 2250*1280

	// output_width_each_dir --> 각 출력 이미지에서 가질 output width, set_rendering_range 함수를 호출해서 얻는다.
	dim3 blocksPerGrid_F((int)ceil((float)output_width_each_dir[0] / (float)twid), (int)ceil((float)(io_config.LF_height / 2) / (float)thei)); // set a shape of the threads-per-block
	dim3 blocksPerGrid_R((int)ceil((float)output_width_each_dir[1] / (float)twid), (int)ceil((float)(io_config.LF_height / 2) / (float)thei)); // set a shape of the threads-per-block
	dim3 blocksPerGrid_B((int)ceil((float)output_width_each_dir[2] / (float)twid), (int)ceil((float)(io_config.LF_height / 2) / (float)thei)); // set a shape of the threads-per-block
	dim3 blocksPerGrid_L((int)ceil((float)output_width_each_dir[3] / (float)twid), (int)ceil((float)(io_config.LF_height / 2) / (float)thei)); // set a shape of the threads-per-block

	synthesize << < blocksPerGrid_F, threadsPerBlock, 0, stream_main >> > (u_synthesized_view, LRU->d_devPtr_hashmap_odd, LRU->d_devPtr_hashmap_even, 0, mode, 0, curPosX, curPosY, io_config.LF_width, io_config.LF_height, io_config.LF_length, io_config.slice_width);
	cudaError_t err = cudaStreamSynchronize(stream_main);
	assert(err == cudaSuccess);
	synthesize << < blocksPerGrid_R, threadsPerBlock, 0, stream_main >> > (u_synthesized_view, LRU->d_devPtr_hashmap_odd, LRU->d_devPtr_hashmap_even, output_width_each_dir[0], mode, 1, curPosX, curPosY, io_config.LF_width, io_config.LF_height, io_config.LF_length, io_config.slice_width);
	err = cudaStreamSynchronize(stream_main);
	assert(err == cudaSuccess);
	synthesize << < blocksPerGrid_B, threadsPerBlock, 0, stream_main >> > (u_synthesized_view, LRU->d_devPtr_hashmap_odd, LRU->d_devPtr_hashmap_even, output_width_each_dir[0] + output_width_each_dir[1], mode, 2, curPosX, curPosY, io_config.LF_width, io_config.LF_height, io_config.LF_length, io_config.slice_width);
	err = cudaStreamSynchronize(stream_main);
	assert(err == cudaSuccess);
	synthesize << < blocksPerGrid_L, threadsPerBlock, 0, stream_main >> > (u_synthesized_view, LRU->d_devPtr_hashmap_odd, LRU->d_devPtr_hashmap_even, output_width_each_dir[0] + output_width_each_dir[1] + output_width_each_dir[2], mode, 3, curPosX, curPosY, io_config.LF_width, io_config.LF_height, io_config.LF_length, io_config.slice_width);
	err = cudaStreamSynchronize(stream_main);
	assert(err == cudaSuccess);

	state_main_thread = MAIN_THREAD_COMPLETE;

	return u_synthesized_view;
}

void LF_Renderer::terminate()
{
	state_main_thread = MAIN_THREAD_TERMINATED;
}

void LF_Renderer::load_slice_set(SliceSet slice_set[][100], std::string prefix) {
	for (int x = 0; x < 100; x++) {
		for (int y = 0; y < 100; y++)
		{
			std::string fname = prefix + std::to_string(x) + "_" + std::to_string(y) + ".txt";
			FILE* fp = fopen(fname.c_str(), "r");

			while (!feof(fp)) {
				int dir, img, pixLn_s, pixLn_e;
				fscanf(fp, "%d\t%d\t%d\t%d\n", &dir, &img, &pixLn_s, &pixLn_e);
				SliceRange sr((FOUR_DIRECTION)dir, img, pixLn_s / io_config.slice_width, pixLn_e / io_config.slice_width);
				slice_set[x][y].push_back(sr);
			}
			fclose(fp);
		}
	}
}

void LF_Renderer::set_rendering_range(int* localPosX, int* localPosY, int* output_width, const int& curPosX, const int& curPosY)
{
	localPosX[0] = curPosX % 100 - 50;
	localPosY[0] = curPosY % 100 - 50;
	localPosX[1] = -1 * localPosY[0];
	localPosY[1] = localPosX[0];
	localPosX[2] = -1 * localPosX[0];
	localPosY[2] = -1 * localPosY[0];
	localPosX[3] = localPosY[0];
	localPosY[3] = -1 * localPosX[0];

	for (int i = 0; i < 4; i++) {
		float theta_L = rad2deg(atan2((-1.0 * LFU_WIDTH / 2 - localPosX[i]), (LFU_WIDTH / 2 - localPosY[i])));
		float theta_R = rad2deg(atan2((1.0 * LFU_WIDTH / 2 - localPosX[i]), (LFU_WIDTH / 2 - localPosY[i])));
		output_width[i] = (int)((theta_R - theta_L) / 0.04f);
	}
}

void LF_Renderer::getNeighborList(std::vector<std::pair<int, int>>& nbrPosition, const int& curPosX, const int& curPosY)
{
	nbrPosition[0] = (std::make_pair(curPosX, curPosY - 1));
	nbrPosition[1] = (std::make_pair(curPosX + 1, curPosY - 1));
	nbrPosition[2] = (std::make_pair(curPosX + 1, curPosY));
	nbrPosition[3] = (std::make_pair(curPosX + 1, curPosY + 1));
	nbrPosition[4] = (std::make_pair(curPosX, curPosY + 1));
	nbrPosition[5] = (std::make_pair(curPosX - 1, curPosY + 1));
	nbrPosition[6] = (std::make_pair(curPosX - 1, curPosY));
	nbrPosition[7] = (std::make_pair(curPosX - 1, curPosY - 1));
}

void LF_Renderer::loop_nbrs_h2d(LRUCache& LRU, const LFU_Window& window, SliceSet slice_set[][100], std::vector<std::pair<int, int>>& nbrPosition, cudaStream_t stream_h2d, H2D_THREAD_STATE& state_h2d_thread, const MAIN_THREAD_STATE& state_main_thread, std::mutex& mtx)
{
	while (state_main_thread != MAIN_THREAD_TERMINATED) {
		mtx.lock();
		cache_slice_in_background(LRU, window, slice_set, nbrPosition, stream_h2d, state_h2d_thread, state_main_thread);
		mtx.unlock();
	}
}

void LF_Renderer::loop_read_disk(LFU_Window& window, const int& prevPosX, const int& prevPosY, const int& curPosX, const int& curPosY, const int& light_field_size, const MAIN_THREAD_STATE& state_main_thread)
{
	while (state_main_thread != MAIN_THREAD_TERMINATED) {
		int ret = window.update_window(prevPosX, prevPosY, curPosX, curPosY, light_field_size, state_main_thread);
		if (ret < 0)
			printf("Neighbor LFs read Interrupted\n");
	}
}

int LF_Renderer::cache_slice(LRUCache& LRU, const LFU_Window& window, SliceSet slice_set[][100], const int& posX, const int& posY) {
	int localPosX = posX % 100;
	int localPosY = posY % 100;

	for (SliceSet::iterator it = slice_set[localPosX][localPosY].begin(); it != slice_set[localPosX][localPosY].end(); it++)
	{
		for (int slice_num = it->range_begin; slice_num <= it->range_end; slice_num++)
		{

			SliceID id;
			id.lf_number = window.m_center->LF[it->direction]->LF_number;
			id.image_number = it->image_num;
			id.slice_number = slice_num;

			int slice_location = find_slice_from_LF(id.image_number, id.slice_number);

			if (window.pinned_memory_status < PINNED_LFU_ODD_AVAILABLE) {
				if (window.m_center->LF[it->direction]->progress == LF_READ_PROGRESS_ODD_FIELD_PREPARED) {
					LRU.put(id, window.m_center->LF[it->direction]->odd_field + slice_location, ODD);
				}
				else if (window.m_center->LF[it->direction]->progress == LF_READ_PROGRESS_EVEN_FIELD_PREPARED) {
					LRU.put(id, window.m_center->LF[it->direction]->odd_field + slice_location, ODD);
					LRU.put(id, window.m_center->LF[it->direction]->even_field + slice_location, EVEN);
				}
				else {
					LRU.enqueue_wait_slice(id, window.m_pinnedLFU[ODD][it->direction] + slice_location, ODD);
					LRU.enqueue_wait_slice(id, window.m_pinnedLFU[EVEN][it->direction] + slice_location, EVEN);
				}
			}
			else if (window.pinned_memory_status == PINNED_LFU_ODD_AVAILABLE) {
				LRU.put(id, window.m_pinnedLFU[ODD][it->direction] + slice_location, ODD);
				LRU.enqueue_wait_slice(id, window.m_pinnedLFU[EVEN][it->direction] + slice_location, EVEN);
			}
			else {
				LRU.put(id, window.m_pinnedLFU[ODD][it->direction] + slice_location, ODD);
				LRU.put(id, window.m_pinnedLFU[EVEN][it->direction] + slice_location, EVEN);
			}
		}
	}

	return 0;
}

int LF_Renderer::cache_slice_in_background(LRUCache& LRU, const LFU_Window& window, SliceSet slice_set[][100], std::vector<std::pair<int, int>>& nbrPosition, cudaStream_t stream_h2d, H2D_THREAD_STATE& thread_state_h2d, const MAIN_THREAD_STATE& thread_state_main) {
	int i = 0;
	int s = 0;

	while (1)
	{
		while (1) {
			for (int p = 0; p < 8; p++) {
				if (thread_state_main == MAIN_THREAD_H2D) {
					thread_state_h2d = H2D_THREAD_INTERRUPTED;
					return -1;
				} // interrupted

				int posX_at_p = nbrPosition.at(p).first % 100;
				int posY_at_p = nbrPosition.at(p).second % 100;


				if (i < slice_set[posX_at_p][posY_at_p].size())
				{
					int slice_num = slice_set[posX_at_p][posY_at_p].at(i).range_begin + s;
					int dir = slice_set[posX_at_p][posY_at_p].at(i).direction;
					int img_num = slice_set[posX_at_p][posY_at_p].at(i).image_num;

					if (slice_num <= slice_set[posX_at_p][posY_at_p].at(i).range_end) {
						SliceID id;
						id.lf_number = window.m_center->LF[dir]->LF_number;
						id.image_number = img_num;
						id.slice_number = slice_num;

						int slice_location = find_slice_from_LF(id.image_number, id.slice_number);
						if (window.pinned_memory_status == PINNED_LFU_ODD_AVAILABLE) {
							LRU.put(id, window.m_pinnedLFU[ODD][dir] + slice_location, stream_h2d, thread_state_h2d, ODD);
						}
						if (window.pinned_memory_status == PINNED_LFU_EVEN_AVAILABLE) {
							LRU.put(id, window.m_pinnedLFU[ODD][dir] + slice_location, stream_h2d, thread_state_h2d, ODD);
							LRU.put(id, window.m_pinnedLFU[EVEN][dir] + slice_location, stream_h2d, thread_state_h2d, EVEN);
						}
					}
				}
			}
			i++;
			if (i >= slice_set[nbrPosition.back().first % 100][nbrPosition.back().second % 100].size()) {
				i = 0;
				break;
			}
		}
		s++;
		if (s > slice_set[nbrPosition.back().first % 100][nbrPosition.back().second % 100].back().range_end) return 0;
	}
}

int LF_Renderer::find_slice_from_LF(const int& img, const int& slice)
{
	return (img * io_config.LF_width * io_config.LF_height + slice * io_config.slice_width * io_config.LF_height) * 3 / 2;
}

__device__ int dev_find_pixel_location(int img, int w, int h, int width, int height, int slice_width)
{
	int slice = w / slice_width;
	int slice_number = w % slice_width;
	return img * width * height * 3 + slice * slice_width * height * 3 + slice_number * height * 3 + h * 3;
}

__device__ int dev_query_hashmap(int lf, int img, int slice, int width, int length, int slice_width)
{
	return lf * (width / slice_width) * length + img * (width / slice_width) + slice;
}

__global__ void synthesize(uint8_t* outImage, uint8_t** d_hashmap_odd, uint8_t** d_hashmap_even, int offset, int mode, int direction, int posX, int posY, int width, int height, int legnth, int slice_width, float fov, float times)
{
	int tw = blockIdx.x * blockDim.x + threadIdx.x; // blockIdx.x = (int)[0, (out_w - 1)]
	int th = blockIdx.y * blockDim.y + threadIdx.y; // threadIdx = (int)[0, (g_height - 1)]

	int localPosX = posX % 100 - 50;
	int localPosY = posY % 100 - 50;

	if (direction == 1) {
		localPosX = -1 * localPosY;
		localPosY = localPosX;
	}
	else if (direction == 2) {
		localPosX = -1 * localPosX;
		localPosY = -1 * localPosY;
	}
	else if (direction == 3) {
		localPosX = localPosY;
		localPosY = -1 * localPosX;
	}

	int LFUW = 100;
	int DATAW = 50;
	int Y = LFUW / 2;

	float theta_L = dev_rad2deg(atan2f((-1.0f * LFUW / 2 - localPosX), (LFUW / 2 - localPosY)));
	float theta_R = dev_rad2deg(atan2f((1.0f * LFUW / 2 - localPosX), (LFUW / 2 - localPosY)));

	int output_width = (int)((theta_R - theta_L) / 0.04f);

	if (tw < output_width) {
		float theta_P = theta_L + (0.04f * (float)tw);

		float b = sqrt(2.0f) * LFUW;
		float xP = (float)(Y - localPosY) * tanf(dev_deg2rad(theta_P)) + localPosX;

		float N_dist = sqrt((float)((xP - localPosX) * (xP - localPosX) + (Y - localPosY) * (Y - localPosY))) / b;

		xP /= 2;
		int P_1 = (int)(roundf(xP + (DATAW >> 1)));
		if (direction == 1 || direction == 2) {
			P_1 = DATAW - P_1 - 1;
		}
		P_1 = dev_Clamp(P_1, 0, DATAW - 1);

		float U = (theta_P * (1.0f / 180.0f)) * (width >> 1) + (width >> 1);
		int U_1 = (int)(roundf(U));
		if (direction == 1) U_1 += width >> 2;
		if (direction == 2) U_1 += width >> 1;
		if (direction == 3) U_1 -= width >> 2;
		if (U_1 >= width) U_1 = U_1 - width;
		else if (U_1 < 0) U_1 = U_1 + width;
		U_1 = dev_Clamp(U_1, 0, width - 1);

		int N_off = (int)(roundf(times * N_dist + 0.5)) >> 1;

		int LF_num = dev_find_LF_number_BMW(direction, posX, posY);
		int image_num = P_1 % legnth;
		int slice_num = U_1 / slice_width;
		int pixel_col = U_1 % slice_width;

		float N_H_r = (float)(HEIGHT + N_off) / height;

		float h_n = (th - height / 2) * N_H_r + height / 2;

		if (h_n < 0)
			h_n = (-1 * h_n) - 1;
		else if (h_n > height - 1)
			h_n = height - ((h_n - height) - 1);

		int H_1 = (int)(roundf(h_n));
		H_1 = dev_Clamp(H_1, 0, height - 1);
		float H_r = h_n - H_1;

		int slice = dev_query_hashmap(LF_num, image_num, slice_num, width, legnth, slice_width); // Random access to hashmap

		uint8_t oddpel_ch0 = d_hashmap_odd[slice][(pixel_col * (height >> 1)) * 3 + H_1 * 3 + 0]; // Random access to pixel column
		uint8_t oddpel_ch1 = d_hashmap_odd[slice][(pixel_col * (height >> 1)) * 3 + H_1 * 3 + 1]; // Random access to pixel column
		uint8_t oddpel_ch2 = d_hashmap_odd[slice][(pixel_col * (height >> 1)) * 3 + H_1 * 3 + 2]; // Random access to pixel column
		outImage[((2 * th) * (9000 * 3) + offset * 3) + tw * 3 + 0] = oddpel_ch0; // b 
		outImage[((2 * th) * (9000 * 3) + offset * 3) + tw * 3 + 1] = oddpel_ch1; // g 
		outImage[((2 * th) * (9000 * 3) + offset * 3) + tw * 3 + 2] = oddpel_ch2; // r 

		if (mode == 1) {
			uint8_t evenpel_ch0 = d_hashmap_even[slice][(pixel_col * (height >> 1)) * 3 + H_1 * 3 + 0]; // Random access to pixel column
			uint8_t evenpel_ch1 = d_hashmap_even[slice][(pixel_col * (height >> 1)) * 3 + H_1 * 3 + 1]; // Random access to pixel column
			uint8_t evenpel_ch2 = d_hashmap_even[slice][(pixel_col * (height >> 1)) * 3 + H_1 * 3 + 2]; // Random access to pixel column

			outImage[((2 * th + 1) * (9000 * 3) + offset * 3) + tw * 3 + 0] = evenpel_ch0; // b 
			outImage[((2 * th + 1) * (9000 * 3) + offset * 3) + tw * 3 + 1] = evenpel_ch1; // g 
			outImage[((2 * th + 1) * (9000 * 3) + offset * 3) + tw * 3 + 2] = evenpel_ch2; // r 
		}
		else
		{
			outImage[((2 * th + 1) * (9000 * 3) + offset * 3) + tw * 3 + 0] = oddpel_ch0; // b 
			outImage[((2 * th + 1) * (9000 * 3) + offset * 3) + tw * 3 + 1] = oddpel_ch1; // g 
			outImage[((2 * th + 1) * (9000 * 3) + offset * 3) + tw * 3 + 2] = oddpel_ch2; // r 
		}
	}
}
