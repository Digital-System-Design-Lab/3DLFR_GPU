#include "LF_Renderer.cuh"

// #define USE_UNIFIED_MEMORY

LF_Renderer::LF_Renderer(const std::string& path_LF, const std::string& path_pixelrange, const size_t& iw, const size_t& ih, const size_t& lf_length, const size_t& num_LFs, const double& dpp, const int& initPosX, const int& initPosY, bool use_window)
{
	size_t output_width = (size_t)(360.0 / dpp);
#ifdef USE_UNIFIED_MEMORY
	synthesized_view = alloc_uint8(output_width * ih * 3, "unified"); // output view를 위한 버퍼
#else
	synthesized_view = alloc_uint8(output_width * ih * 3, "device"); // output view를 위한 버퍼
#endif
	
	generate_LFConfig(path_LF, path_pixelrange, iw, ih, lf_length, num_LFs, dpp);
	LRU_Cache* lru_cache = new LRU_Cache(this->config, &state_h2d_thread); // LRU 캐시 초기화 (LF 개수, 캐싱할 원소 개수)
	this->LRU = lru_cache;

	this->use_window = use_window;
	const size_t light_field_size = (this->config->LF_width * this->config->LF_height * this->config->LF_length * 3) / 2; // 라이트 필드 파일 사이즈

	curPosX = initPosX; // 초기 위치
	curPosY = initPosY;
	prevPosX = curPosX;
	prevPosY = curPosY;
	this->nbrPosition.resize(8); // 현재 viewpoint의 8개 이웃 
	getNeighborList(nbrPosition, curPosX, curPosY);

	load_slice_set(this->slice_set, config->path_PixelRange); // ./PixelRange의 파일을 읽고 pixel range를 load

	LFU_Window* lfu_window;
	if (this->use_window == true)
		lfu_window = new LFU_Window(this->config, curPosX, curPosY, &state_disk_read_thread); // 3x3 형태의 LFU Window, 2D listaaa 
	else
		lfu_window = new LFU_Window(this->config, curPosX, curPosY, &state_disk_read_thread, false); // 3x3 형태의 LFU Window, 2D list
	this->window = lfu_window;

	state_main_thread = MAIN_THREAD_INIT; // 백그라운드 Disk->Hostmem LF read를 담당하는 초기 스레드 상태
	state_h2d_thread = H2D_THREAD_INIT; // 백그라운드 LRU 캐싱을 담당하는 스레드 초기상태
	state_disk_read_thread = DISK_READ_THREAD_NEIGHBOR_LFU_READ_COMPLETE;

	cudaStreamCreate(&stream_main); // CUDA Concurrency를 위한 streams
	cudaStreamCreate(&stream_h2d);
	// workers.push_back(std::thread(&LF_Renderer::loop_nbrs_h2d, this, std::ref(*LRU), std::ref(*window), slice_set, std::ref(nbrPosition), stream_h2d, std::ref(state_main_thread), std::ref(mtx)));
	// H2D, LF Read를 위한 worker threads
	if (this->use_window == true) {
		workers.push_back(std::thread(&LF_Renderer::loop_nbrs_h2d, this, std::ref(*LRU), std::ref(*window), slice_set, std::ref(nbrPosition), stream_h2d, std::ref(state_main_thread), std::ref(mtx)));
		workers.push_back(std::thread(&LF_Renderer::loop_read_disk, this, std::ref(*window), std::ref(curPosX), std::ref(curPosY), std::ref(light_field_size), std::ref(state_main_thread)));
	}

	threadsPerBlock.x = 4;
	threadsPerBlock.y = 64;

	query_CudaMemory();
}

LF_Renderer::~LF_Renderer() {
	printf("Destruct LF Renderer\n");
	for (std::vector<std::thread>::iterator it = workers.begin(); it != workers.end(); it++) {
		it->join();
	}
	delete this->LRU;
	delete this->window;
#ifdef USE_UNIFIED_MEMORY
	free_uint8(synthesized_view, "unified");
#else
	free_uint8(synthesized_view, "device");
#endif
	cudaStreamDestroy(stream_main);
	cudaStreamDestroy(stream_h2d);
	delete this->config;
}

// 렌더링 함수 
uint8_t* LF_Renderer::do_rendering(int& newPosX, int& newPosY)
{
	StopWatch sw;
	sw.Start();
	if (this->use_window == false) {
		if (getLFUID(curPosX, curPosY) != getLFUID(newPosX, newPosY)) {
			printf("out-of-renderable range, return previous view\n");

			newPosX = curPosX;
			newPosY = curPosY;

			return this->synthesized_view;
		}
		else {
			this->prevPosX = curPosX;
			this->prevPosY = curPosY;
			this->curPosX = newPosX;
			this->curPosY = newPosY;
		}
	}
	else {
		this->prevPosX = curPosX;
		this->prevPosY = curPosY;
		this->curPosX = newPosX;
		this->curPosY = newPosY;

		curPosX = clamp(curPosX, 101, 499);
		curPosY = clamp(curPosY, 101, 5499);
	}

	set_rendering_params(localPosX, localPosY, output_width_each_dir, curPosX, curPosY); // CUDA 블록 사이즈 설정, 렌더링할 범위 설정 등

	state_main_thread = MAIN_THREAD_WAIT;
	getNeighborList(nbrPosition, curPosX, curPosY); // viewpoint의 이웃 8개를 리턴

	while (!(getLFUID(curPosX, curPosY) == window->m_center->id && state_disk_read_thread >= DISK_READ_THREAD_CENTER_LFU_READ_COMPLETE)) {}
	printf("[%d] (%d, %d)\n", getLFUID(curPosX, curPosY), curPosX, curPosY);

	state_main_thread = MAIN_THREAD_H2D;
	mtx.lock();
	std::pair<int, int> hit_per_slice = cache_slice(*LRU, *window, slice_set, curPosX, curPosY); // 현재 viewpoint에 필요한 slices를 캐싱

	int mode = LRU->synchronize_HashmapOfPtr(*window, stream_main); // Host memory <-> Device memory 동기화
	mtx.unlock();
	double caching_time = sw.Stop();
	printf("caching time : %f\n", caching_time);
	
	cache_validity_check(curPosX, curPosY, slice_set, window, LRU);
	sw.Start();
	state_main_thread = MAIN_THREAD_RENDERING;
	// interlace mode -> block shape : 2250*1280
	
	// output_width_each_dir --> 각 출력 이미지에서 가질 output width, set_rendering_range 함수를 호출해서 얻는다.
	// printf("output width : %d = L%d + F%d + R%d + B%d\n", output_width, output_width_each_dir[3], output_width_each_dir[0], output_width_each_dir[1], output_width_each_dir[2]);
	// launch rendering kernel
	cudaError_t err;
	synthesize << < blocksPerGrid_L, threadsPerBlock, 0, stream_main >> > (synthesized_view, LRU->d_devPtr_hashmap_odd, LRU->d_devPtr_hashmap_even, 0, mode, 3, curPosX, curPosY, localPosX[3], localPosY[3], (int)config->LF_width, (int)config->LF_height, (int)config->LF_length, (int)config->slice_width);
	synthesize << < blocksPerGrid_F, threadsPerBlock, 0, stream_main >> > (synthesized_view, LRU->d_devPtr_hashmap_odd, LRU->d_devPtr_hashmap_even, output_width_each_dir[3], mode, 0, curPosX, curPosY, localPosX[0], localPosY[0], (int)config->LF_width, (int)config->LF_height, (int)config->LF_length, (int)config->slice_width);
	synthesize << < blocksPerGrid_R, threadsPerBlock, 0, stream_main >> > (synthesized_view, LRU->d_devPtr_hashmap_odd, LRU->d_devPtr_hashmap_even, output_width_each_dir[3] + output_width_each_dir[0], mode, 1, curPosX, curPosY, localPosX[1], localPosY[1], (int)config->LF_width, (int)config->LF_height, (int)config->LF_length, (int)config->slice_width);
	synthesize << < blocksPerGrid_B, threadsPerBlock, 0, stream_main >> > (synthesized_view, LRU->d_devPtr_hashmap_odd, LRU->d_devPtr_hashmap_even, output_width_each_dir[3] + output_width_each_dir[0] + output_width_each_dir[1], mode, 2, curPosX, curPosY, localPosX[2], localPosY[2], (int)config->LF_width, (int)config->LF_height, (int)config->LF_length, (int)config->slice_width);
	err = cudaStreamSynchronize(stream_main);
	assert(err == cudaSuccess);
	state_main_thread = MAIN_THREAD_COMPLETE;

	double rendering_time = sw.Stop();
	printf("rendering time : %f\n", rendering_time);
	
	FILE* fh2d = fopen(("./experiments/prefetching/npf_" + std::to_string(this->config->slice_width) + ".log").c_str(), "a");
	if(curPosX !=201)
		fprintf(fh2d, "%d,%d\t%f\t%f\t%d\t%d\n", curPosX, curPosY, caching_time, rendering_time, hit_per_slice.first, hit_per_slice.second);
	fclose(fh2d);

	return synthesized_view;
}

// 종료 함수 (worker threads join을 위해)
void LF_Renderer::terminate()
{
	state_main_thread = MAIN_THREAD_TERMINATED;
}

void LF_Renderer::load_slice_set(SliceSet slice_set[][100], std::string prefix)
{
	for (int x = 0; x < 100; x++) {
		for (int y = 0; y < 100; y++)
		{
			std::string fname = prefix + std::to_string(x) + "_" + std::to_string(y) + ".txt";
			FILE* fp = fopen(fname.c_str(), "r");

			while (!feof(fp)) {
				int dir, img, pixLn_s, pixLn_e;
				fscanf(fp, "%d\t%d\t%d\t%d\n", &dir, &img, &pixLn_s, &pixLn_e);
				SliceRange sr((FOUR_DIRECTION)dir, img, pixLn_s / config->slice_width, pixLn_e / config->slice_width);
				slice_set[x][y].push_back(sr);
			}
			fclose(fp);
		}
	}
}

void LF_Renderer::set_rendering_params(int* localPosX, int* localPosY, int* output_width, const int& curPosX, const int& curPosY)
{
	localPosX[0] = (curPosX % 100) - 50;
	localPosY[0] = (curPosY % 100) - 50;
	localPosX[1] = -1 * localPosY[0];
	localPosY[1] = localPosX[0];
	localPosX[2] = -1 * localPosX[0];
	localPosY[2] = -1 * localPosY[0];
	localPosX[3] = localPosY[0];
	localPosY[3] = -1 * localPosX[0];

	for (int i = 0; i < 4; i++) {
		float theta_L = rad2deg(atan2f((-1.0f * LFU_WIDTH / 2 - localPosX[i]), (LFU_WIDTH / 2 - localPosY[i])));
		float theta_R = rad2deg(atan2f((1.0f * LFU_WIDTH / 2 - localPosX[i]), (LFU_WIDTH / 2 - localPosY[i])));
		output_width[i] = (int)((theta_R - theta_L) / 0.04f);
	}

	blocksPerGrid_F.x = (int)ceil((float)output_width[0] / (float)threadsPerBlock.x);
	blocksPerGrid_R.x = (int)ceil((float)output_width[1] / (float)threadsPerBlock.x);
	blocksPerGrid_B.x = (int)ceil((float)output_width[2] / (float)threadsPerBlock.x);
	blocksPerGrid_L.x = (int)ceil((float)output_width[3] / (float)threadsPerBlock.x);
	blocksPerGrid_F.y = (int)ceil((float)(config->LF_height / 2) / (float)threadsPerBlock.y); // set a shape of the threads-per-block
	blocksPerGrid_R.y = (int)ceil((float)(config->LF_height / 2) / (float)threadsPerBlock.y); // set a shape of the threads-per-block
	blocksPerGrid_B.y = (int)ceil((float)(config->LF_height / 2) / (float)threadsPerBlock.y); // set a shape of the threads-per-block
	blocksPerGrid_L.y = (int)ceil((float)(config->LF_height / 2) / (float)threadsPerBlock.y); // set a shape of the threads-per-block
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

void LF_Renderer::loop_nbrs_h2d(LRU_Cache& LRU, const LFU_Window& window, SliceSet slice_set[][100], std::vector<std::pair<int, int>>& nbrPosition, cudaStream_t stream_h2d, const MAIN_THREAD_STATE& state_main_thread, std::mutex& mtx)
{
	while (state_main_thread != MAIN_THREAD_TERMINATED) {
		mtx.lock();
		cache_slice_in_background(LRU, window, slice_set, nbrPosition, stream_h2d, state_main_thread);
		mtx.unlock();
	}
}

void LF_Renderer::loop_read_disk(LFU_Window& window, const int& curPosX, const int& curPosY, const int& light_field_size, const MAIN_THREAD_STATE& state_main_thread)
{
	while (state_main_thread != MAIN_THREAD_TERMINATED) {
		int ret = window.update_window(curPosX, curPosY, light_field_size, state_main_thread);
		if (ret < 0)
			printf("Neighbor LFs read Interrupted\n");
	}
}

std::pair<int, int> LF_Renderer::cache_slice(LRU_Cache& LRU, const LFU_Window& window, SliceSet slice_set[][100], const int& posX, const int& posY)
{
	int localPosX = posX % 100;
	int localPosY = posY % 100;

	int hit_count = 0;
	int slice_count = 0;

	for (SliceSet::iterator it = slice_set[localPosX][localPosY].begin(); it != slice_set[localPosX][localPosY].end(); it++)
	{
		for (int slice_num = it->range_begin; slice_num <= it->range_end; slice_num++)
		{
			SliceID id;
			id.lf_number = window.m_center->LF[it->direction]->LF_number;
			id.image_number = it->image_num;
			id.slice_number = slice_num;

			size_t slice_location = find_slice_from_LF(id.image_number, id.slice_number);

			if (window.pinned_memory_status == PINNED_LFU_NOT_AVAILABLE) {
				if (window.m_center->LF[it->direction]->progress == LF_READ_PROGRESS_ODD_FIELD_PREPARED) {
					/* calculate hit_rate */
					int hit = LRU.put(id, window.m_center->LF[it->direction]->odd_field + slice_location, ODD);
					hit_count += hit;
					slice_count += 1;
					/* calculate hit_rate - end*/
				}
				else if (window.m_center->LF[it->direction]->progress == LF_READ_PROGRESS_EVEN_FIELD_PREPARED) {
					int hit = LRU.put(id, window.m_center->LF[it->direction]->odd_field + slice_location, ODD);
					LRU.put(id, window.m_center->LF[it->direction]->even_field + slice_location, EVEN);
					hit_count += hit;
					slice_count += 1;
				}
				else {
					LRU.enqueue_wait_slice(id, window.m_center->LF[it->direction]->odd_field + slice_location, ODD);
					LRU.enqueue_wait_slice(id, window.m_center->LF[it->direction]->even_field + slice_location, EVEN);
				}
			}
			else if (window.pinned_memory_status == PINNED_LFU_ODD_AVAILABLE) {
				int hit = LRU.put(id, window.m_pinnedLFU[ODD][it->direction] + slice_location, ODD);
				LRU.enqueue_wait_slice(id, window.m_pinnedLFU[EVEN][it->direction] + slice_location, EVEN);
				hit_count += hit;
				slice_count += 1;
			}
			else if (window.pinned_memory_status == PINNED_LFU_EVEN_AVAILABLE) {
				int hit = LRU.put(id, window.m_pinnedLFU[ODD][it->direction] + slice_location, ODD);
				LRU.put(id, window.m_pinnedLFU[EVEN][it->direction] + slice_location, EVEN);
				hit_count += hit;
				slice_count += 1;
			}
		}
	}

	return std::make_pair(hit_count, slice_count);
}

int LF_Renderer::cache_slice_in_background(LRU_Cache& LRU, const LFU_Window& window, SliceSet slice_set[][100], std::vector<std::pair<int, int>>& nbrPosition, cudaStream_t stream_h2d, const MAIN_THREAD_STATE& thread_state_main)
{
	int i = 0;
	int s = 0;

	while (1)
	{
		while (1) {
			for (int p = 0; p < 8; p++) {
				if (thread_state_main == MAIN_THREAD_H2D) {
					state_h2d_thread = H2D_THREAD_INTERRUPTED;
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

						size_t slice_location = find_slice_from_LF(id.image_number, id.slice_number);
						if (window.pinned_memory_status == PINNED_LFU_ODD_AVAILABLE) {
							LRU.put(id, window.m_pinnedLFU[ODD][dir] + slice_location, stream_h2d, ODD);
						}
						if (window.pinned_memory_status == PINNED_LFU_EVEN_AVAILABLE) {
							LRU.put(id, window.m_pinnedLFU[ODD][dir] + slice_location, stream_h2d, ODD);
							LRU.put(id, window.m_pinnedLFU[EVEN][dir] + slice_location, stream_h2d, EVEN);
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

size_t LF_Renderer::find_slice_from_LF(const int& img, const int& slice)
{
	return (img * config->LF_width * config->LF_height + slice * config->slice_width * config->LF_height) * 3 / 2;
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

__global__ void synthesize(uint8_t* outImage, uint8_t** d_hashmap_odd, uint8_t** d_hashmap_even, int offset, int mode, int direction, int posX, int posY, int localPosX, int localPosY, int width, int height, int legnth, int slice_width, float fov, float times)
{
	int tw = blockIdx.x * blockDim.x + threadIdx.x; // blockIdx.x = (int)[0, (out_w - 1)]
	int th = blockIdx.y * blockDim.y + threadIdx.y; // threadIdx = (int)[0, (g_height - 1)]

	int LFUW = 100;
	int DATAW = 50;
	int Y = LFUW / 2;

	float theta_L = dev_rad2deg(atan2f((-1.0f * LFUW / 2 - localPosX), (LFUW / 2 - localPosY)));
	float theta_R = dev_rad2deg(atan2f((1.0f * LFUW / 2 - localPosX), (LFUW / 2 - localPosY)));
	if (localPosY == 50) {
		theta_L = -90.0f;
		theta_R = 90.0f;
	}

	int output_width = (int)((theta_R - theta_L) / 0.04f);

	if (tw < output_width && th < (height >> 1)) // Thread index must not exceed output resolution
	{
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

		float N_H_r = (float)(height + N_off) / height;

		float h_n = (th - height / 2) * N_H_r + height / 2;

		if (h_n < 0)
			h_n = (-1 * h_n) - 1;
		else if (h_n > height - 1)
			h_n = height - ((h_n - height) - 1);

		int H_1 = (int)(roundf(h_n));
		H_1 = dev_Clamp(H_1, 0, height - 1);
		float H_r = h_n - H_1;

		int slice = dev_query_hashmap(LF_num, image_num, slice_num, width, legnth, slice_width); // Random access to hashmap

		uint8_t oddpel_ch0 = d_hashmap_odd[slice][(pixel_col * (height >> 1)) * 3 + H_1 * 3 + 2]; // Random access to pixel column
		uint8_t oddpel_ch1 = d_hashmap_odd[slice][(pixel_col * (height >> 1)) * 3 + H_1 * 3 + 1]; // Random access to pixel column
		uint8_t oddpel_ch2 = d_hashmap_odd[slice][(pixel_col * (height >> 1)) * 3 + H_1 * 3 + 0]; // Random access to pixel column
		outImage[((2 * th) * (9000 * 3) + offset * 3) + tw * 3 + 0] = oddpel_ch0; // b 
		outImage[((2 * th) * (9000 * 3) + offset * 3) + tw * 3 + 1] = oddpel_ch1; // g 
		outImage[((2 * th) * (9000 * 3) + offset * 3) + tw * 3 + 2] = oddpel_ch2; // r 

		if (mode == 1) {
			uint8_t evenpel_ch0 = d_hashmap_even[slice][(pixel_col * (height >> 1)) * 3 + H_1 * 3 + 2]; // Random access to pixel column
			uint8_t evenpel_ch1 = d_hashmap_even[slice][(pixel_col * (height >> 1)) * 3 + H_1 * 3 + 1]; // Random access to pixel column
			uint8_t evenpel_ch2 = d_hashmap_even[slice][(pixel_col * (height >> 1)) * 3 + H_1 * 3 + 0]; // Random access to pixel column

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
#if 0
		if (tw == 0) // for debug
		{
			outImage[((2 * th) * (9000 * 3) + offset * 3) + tw * 3 + 0] = 0; // b 
			outImage[((2 * th) * (9000 * 3) + offset * 3) + tw * 3 + 1] = 0; // g 
			outImage[((2 * th) * (9000 * 3) + offset * 3) + tw * 3 + 2] = 255; // r 
			outImage[((2 * th + 1) * (9000 * 3) + offset * 3) + tw * 3 + 0] = 0; // b 
			outImage[((2 * th + 1) * (9000 * 3) + offset * 3) + tw * 3 + 1] = 0; // g 
			outImage[((2 * th + 1) * (9000 * 3) + offset * 3) + tw * 3 + 2] = 255; // r
		}
		if (tw == output_width - 1) // for debug
		{
			outImage[((2 * th) * (9000 * 3) + offset * 3) + tw * 3 + 0] = 255; // b 
			outImage[((2 * th) * (9000 * 3) + offset * 3) + tw * 3 + 1] = 0; // g 
			outImage[((2 * th + 1) * (9000 * 3) + offset * 3) + tw * 3 + 0] = 255; // b 
			outImage[((2 * th + 1) * (9000 * 3) + offset * 3) + tw * 3 + 1] = 0; // g 
		}
#endif
	}
}

// function for debugging
void cache_validity_check(int curPosX, int curPosY, SliceSet slice_set[][100], LFU_Window* window, LRU_Cache* LRU)
{
	bool error = false;
	int required_slices = slice_set[curPosX % 100][curPosY % 100].size();
	printf("%d slices are required\n", required_slices);
	for (SliceSet::iterator ss = slice_set[curPosX % 100][curPosY % 100].begin(); ss != slice_set[curPosX % 100][curPosY % 100].end(); ss++)
	{
		for (int sn = ss->range_begin; sn <= ss->range_end; sn++)
		{
			SliceID test_id;
			test_id.lf_number = window->m_center->LF[ss->direction]->LF_number;
			test_id.image_number = ss->image_num;
			test_id.slice_number = sn;
			
// 			int hasmap_idx = test_id.lf_number * (LF_width / config->slice_width) * config->LF_length + test_id.image_number * (config->LF_width / config->slice_width) + test_id.slice_number;

			uint8_t* test_addr = LRU->find_slice_in_hashmap(test_id);
			if (test_addr == nullptr) {
				printf("[%d-%d-%d] is not exist\n", test_id.lf_number, test_id.image_number, test_id.slice_number);
				error = true;
			}
			else {
				// printf("[%d-%d-%d] test_addr : %x\n", test_id.lf_number, test_id.image_number, test_id.slice_number, test_addr);
				// uint8_t* test_val = LRU->hashmap_odd[LRU->query_hashmap(test_id, ODD)]->odd_data; // back data에서 까만 줄?
			}
		}
	}
	if (error) exit(1);
}

void LF_Renderer::generate_LFConfig(const std::string& path_LF, const std::string& path_pixelrange, const size_t& iw, const size_t& ih, const size_t& lf_len, const size_t& numLFs, const double& dpp)
{
	size_t dm = (size_t)((double)get_devmem_freespace() * 0.8);
	size_t max_pixel_range = (size_t)ceil(2.0 * rad2deg(atan2(0.5, 1.0)) * iw / 360.0);
	size_t neareast_mpr;
	size_t sw_upper_bound = 0;
	double mindiff = 1e6;

	for (int n = 1; n <= iw; n++) {
		if (iw % n == 0) {
			size_t sw = iw / n; // assume a slice width

			if (abs(max_pixel_range - (double)sw) < mindiff)
			{
				mindiff = abs(max_pixel_range - (double)sw);
				neareast_mpr = sw;
			}

			size_t dm_hashmap = (iw / sw) * lf_len * numLFs * sizeof(uint8_t*);
			size_t dm_slice = dm - dm_hashmap - (size_t)(360.0 /dpp) * ih * 3;
			size_t cache_capacity = dm_slice / (sw * ih * 3);
			if (cache_capacity > (4 * lf_len) * 1.2)
				sw_upper_bound = sw_upper_bound < sw ? sw : sw_upper_bound;
		}
	}

	size_t optiman_sw = std::min(neareast_mpr, sw_upper_bound);
	// optiman_sw = 480; // 강제변환

	this->config = new LF_Config(path_LF, path_pixelrange, iw, ih, lf_len, numLFs, optiman_sw, dpp);
}