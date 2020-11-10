#include "LRU_Cache.h"
#include "LFU_Window.h"
#include "BMW_FilePath.h"

#include <thread> // std::thread
#include <future> // std::future
#include <stdlib.h> // size_t
#include <stdint.h> // uint8_t
#define LOGGER 1

__device__ int dev_find_pixel_location(int img, int w, int h, int g_width, int g_height, int g_slice_width)
{
	int slice = w / g_slice_width;
	int slice_number = w % g_slice_width;
	return img * g_width * g_height * 3 + slice * g_slice_width * g_height * 3 + slice_number * g_height * 3 + h * 3;
}

__device__ int dev_query_hashmap(const int& lf, const int& img, const int& slice)
{
	return lf * (g_width / g_slice_width) * g_length + img * (g_width / g_slice_width) + slice;
}

__global__ void rendering(uint8_t* outImage, uint8_t** d_hashmap_odd, uint8_t** d_hashmap_even, int offset, int mode, int direction, int posX, int posY, int g_width, int g_height, int g_slice_width, float fov = 90.0f, float times = 270.0f)
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

	float theta_L = dev_rad2deg(atan2f((-1.0f * LFUW / 2 - localPosX), (LFUW / 2 - localPosY)));
	float theta_R = dev_rad2deg(atan2f((1.0f * LFUW / 2 - localPosX), (LFUW / 2 - localPosY)));

	int output_width = (int)((theta_R - theta_L) / 0.04f);
	
	if (tw < output_width) {
		int Y = LFUW / 2;

		float theta_P = theta_L + (0.04 * (float)tw);
		
		float b = sqrt(2.0) * LFUW;
		float xP = (Y - localPosY) * __tanf(dev_deg2rad(theta_P)) + localPosX;

		float N_dist = sqrt((float)((xP - localPosX) * (xP - localPosX) + (Y - localPosY) * (Y - localPosY))) / b;

		xP /= 2;
		int P_1 = (int)(roundf(xP + (DATAW >> 1)));
		if (direction == 1 || direction == 2) {
			P_1 = DATAW - P_1 - 1;
		}
		P_1 = dev_Clamp(P_1, 0, DATAW - 1);

		float U = (theta_P * (1.0f / 180.0f)) * (WIDTH >> 1) + (WIDTH >> 1);
		int U_1 = (int)(roundf(U));
		if (direction == 1) U_1 += WIDTH >> 2;
		if (direction == 2) U_1 += WIDTH >> 1;
		if (direction == 3) U_1 -= WIDTH >> 2;

		int N_off = (int)(roundf(times * N_dist + 0.5)) >> 1;

		if (U_1 >= WIDTH) U_1 = U_1 - WIDTH;
		else if (U_1 < 0) U_1 = U_1 + WIDTH;
		U_1 = dev_Clamp(U_1, 0, WIDTH - 1);

		int LF_num = dev_find_LF_number_BMW(direction, posX, posY);
		int image_num = P_1 % LENGTH;
		int slice_num = U_1 / g_slice_width;
		int pixel_col = U_1 % g_slice_width;

		float N_H_r = (float)(HEIGHT + N_off) / HEIGHT;

		float h_n = (th - HEIGHT / 2) * N_H_r + HEIGHT / 2;

		if (h_n < 0)
			h_n = (-1 * h_n) - 1;
		else if (h_n > HEIGHT - 1)
			h_n = HEIGHT - ((h_n - HEIGHT) - 1);

		int H_1 = (int)(roundf(h_n));
		H_1 = dev_Clamp(H_1, 0, HEIGHT - 1);
		float H_r = h_n - H_1;

		int slice = dev_query_hashmap(LF_num, image_num, slice_num); // Random access to hashmap
		// printf("%d %d %d(%d) %d, %d %d, %d\n", LF_num, image_num, slice_num, U_1, pixel_col, tw, th, slice);
		
		uint8_t oddpel_ch0 = d_hashmap_odd[slice][(pixel_col * g_height / 2) * 3 + H_1 * 3 + 0]; // Random access to pixel column
		uint8_t oddpel_ch1 = d_hashmap_odd[slice][(pixel_col * g_height / 2) * 3 + H_1 * 3 + 1]; // Random access to pixel column
		uint8_t oddpel_ch2 = d_hashmap_odd[slice][(pixel_col * g_height / 2) * 3 + H_1 * 3 + 2]; // Random access to pixel column
		outImage[((2 * th) * (9000 * 3) + offset * 3) + tw * 3 + 0] = oddpel_ch0; // b 
		outImage[((2 * th) * (9000 * 3) + offset * 3) + tw * 3 + 1] = oddpel_ch1; // g 
		outImage[((2 * th) * (9000 * 3) + offset * 3) + tw * 3 + 2] = oddpel_ch2; // r 

		if (mode == 1) {
			uint8_t evenpel_ch0 = d_hashmap_even[slice][(pixel_col * g_height / 2) * 3 + H_1 * 3 + 0]; // Random access to pixel column
			uint8_t evenpel_ch1 = d_hashmap_even[slice][(pixel_col * g_height / 2) * 3 + H_1 * 3 + 1]; // Random access to pixel column
			uint8_t evenpel_ch2 = d_hashmap_even[slice][(pixel_col * g_height / 2) * 3 + H_1 * 3 + 2]; // Random access to pixel column
		
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

void load_slice_set(SliceSet slice_set[][100]) {
	for (int x = 0; x < 100; x++) {
		for (int y = 0; y < 100; y++)
		{
			std::string fname = "S:/4K/LFU/" + std::to_string(x) + "_" + std::to_string(y) + ".txt";
			FILE* fp = fopen(fname.c_str(), "r");

			while (!feof(fp)) {
				int dir, img, pixLn_s, pixLn_e;
				fscanf(fp, "%d\t%d\t%d\t%d\n", &dir, &img, &pixLn_s, &pixLn_e);
				SliceRange sr((FOUR_DIRECTION)dir, img, pixLn_s / g_slice_width, pixLn_e / g_slice_width);
				slice_set[x][y].push_back(sr);
			}
			fclose(fp);
		}
	}
}

void getNeighborList(std::vector<std::pair<int, int>>& nbrPosition, int curPosX, int curPosY)
{
	nbrPosition.at(0) = (std::make_pair(curPosX, curPosY - 1));
	nbrPosition.at(1) = (std::make_pair(curPosX + 1, curPosY - 1));
	nbrPosition.at(2) = (std::make_pair(curPosX + 1, curPosY));
	nbrPosition.at(3) = (std::make_pair(curPosX + 1, curPosY + 1));
	nbrPosition.at(4) = (std::make_pair(curPosX, curPosY + 1));
	nbrPosition.at(5) = (std::make_pair(curPosX - 1, curPosY + 1));
	nbrPosition.at(6) = (std::make_pair(curPosX - 1, curPosY));
	nbrPosition.at(7) = (std::make_pair(curPosX - 1, curPosY - 1));
}

void set_both_end_image(int& leftend_image, int& rightend_image, const int& posX, const int& posY)
{
	leftend_image = (posX - posY);
	rightend_image = (posX + posY);
	if (leftend_image < 0 || posY <= 0 || posX <= 0) {
		printf("OUT OF RENDERABLE AREA\n");
		exit(1);
	}
}

std::pair<size_t, size_t> cache_slice(LRUCache& LRU, const LFU_Window& window, SliceSet slice_set[][100], const int& posX, const int& posY) {

	size_t hit = 0;
	size_t try_caching = 0;

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

			int slice_location = find_slice_from_LF(id.image_number, id.slice_number, true);

			if (window.pinned_memory_status < PINNED_LFU_ODD_AVAILABLE) {
				LRU.enqueue_wait_slice(id, window.m_pinnedLFU[ODD][it->direction] + slice_location, ODD);
				LRU.enqueue_wait_slice(id, window.m_pinnedLFU[EVEN][it->direction] + slice_location, EVEN);
			}
			else if (window.pinned_memory_status == PINNED_LFU_ODD_AVAILABLE) {
				LRU.put(id, window.m_pinnedLFU[ODD][it->direction] + slice_location, ODD);
				LRU.enqueue_wait_slice(id, window.m_pinnedLFU[EVEN][it->direction] + slice_location, EVEN);
			}
			else {
				LRU.put(id, window.m_pinnedLFU[ODD][it->direction] + slice_location, ODD);
				LRU.put(id, window.m_pinnedLFU[EVEN][it->direction] + slice_location, EVEN);
			}
			try_caching++;
		}
	}

	return std::make_pair(hit, try_caching);
}

int cache_slice_in_background(LRUCache& LRU, const LFU_Window& window, SliceSet slice_set[][100], std::vector<std::pair<int, int>>& nbrPosition, cudaStream_t stream_h2d, H2D_THREAD_STATE& thread_state_h2d, const MAIN_THREAD_STATE& thread_state_main) {
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

						int slice_location = find_slice_from_LF(id.image_number, id.slice_number, true);
						if (window.pinned_memory_status == PINNED_LFU_ODD_AVAILABLE) {
							if (!LRU.isFull(ODD))
								LRU.put(id, window.m_pinnedLFU[ODD][dir] + slice_location, stream_h2d, thread_state_h2d, ODD);
						}
						if (window.pinned_memory_status == PINNED_LFU_EVEN_AVAILABLE) {
							if (!LRU.isFull(ODD))
								LRU.put(id, window.m_pinnedLFU[ODD][dir] + slice_location, stream_h2d, thread_state_h2d, ODD);
							if (!LRU.isFull(EVEN))
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

void loop_nbrs_h2d(LRUCache& LRU, const LFU_Window& window, SliceSet slice_set[][100], std::vector<std::pair<int, int>>& nbrPosition, cudaStream_t stream_h2d, H2D_THREAD_STATE& thread_state_h2d, const MAIN_THREAD_STATE& thread_state_main, std::mutex& mtx)
{
	bool loop = true;
	while (loop) {
		mtx.lock();
		cache_slice_in_background(LRU, window, slice_set, nbrPosition, stream_h2d, thread_state_h2d, thread_state_main);
		mtx.unlock();
		if (thread_state_main == MAIN_THREAD_TERMINATED) loop = false;
	}
}

void loop_read_disk(LFU_Window& window, const int& prevPosX, const int& prevPosY, const int& curPosX, const int& curPosY, const int& light_field_size, READ_DISK_THREAD_STATE& read_disk_thread_state, const MAIN_THREAD_STATE& main_thread_state)
{
	bool loop = true;
	while (loop) {
		window.update_window(prevPosX, prevPosY, curPosX, curPosY, light_field_size, main_thread_state);
		if (main_thread_state == MAIN_THREAD_TERMINATED) loop = false;
	}
}

void set_rendering_range(int* localPosX, int* localPosY, int* output_width, const int& curPosX, const int& curPosY)
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

int main()
{
	/* Declare */
	StopWatch sw; // for benchmark
	const size_t limit_cached_slice = (size_t)ceil((size_t)1.6 * (size_t)1024 * (size_t)1024 * (size_t)1024 / g_slice_size);
	const size_t limit_hashing_LF = 71;

	printf("Input resolution : %dx%dx%d\n", g_width, g_height, g_length);
	printf("Output resolution : %dx%d\n", g_output_width, g_height);
	printf("Slice resolution : %dx%d\n", g_slice_width, g_height);
	printf("Slice Cache Size Limit : %llu items -> %lf MB\n", limit_cached_slice, g_slice_size * limit_cached_slice / 1e6 * 2);
	printf("Hashing LF Range Limit : %d to %d\n", 0, limit_hashing_LF);

	const size_t light_field_size = g_width * g_height * g_length * 3 / 2;

	int localPosX[4];
	int localPosY[4];
	int output_width_each_dir[4];

	int curPosX = 201;
	int curPosY = 201;
	int prevPosX = curPosX;
	int prevPosY = curPosY;

	std::vector<std::pair<int, int>> nbrPosition(8);

	/* Initialize */
	SliceSet slice_set[100][100];
	load_slice_set(slice_set);

	cudaStream_t stream_main, stream_h2d;
	cudaStreamCreate(&stream_main);
	cudaStreamCreate(&stream_h2d);

	LRUCache LRU(limit_hashing_LF, limit_cached_slice);
	printf("[MAIN] LRU Cache is created\n");
	LFU_Window window(curPosX, curPosY, light_field_size);
	printf("[MAIN] LFU Window is created\n");
	uint8_t* u_synthesized_view;
	u_synthesized_view = alloc_uint8(g_output_width * g_height * 3, "unified");
	printf("[MAIN] Memory space for output is allocated\n");

	MAIN_THREAD_STATE state_main_thread;
	H2D_THREAD_STATE state_h2d_thread;
	READ_DISK_THREAD_STATE state_read_thread;

	state_main_thread = MAIN_THREAD_INIT;
	state_h2d_thread = H2D_THREAD_INIT;
	state_read_thread = READ_DISK_THREAD_NEIGHBOR_LF_READ_COMPLETE;
	 
	int dir = 0;
	
	// for result analysis
	std::vector<double> time_end_to_end;
	std::vector<std::pair<size_t, size_t>> reused_per_total;
	std::vector<int> field_mode;
	std::vector<std::pair<int, int>> position_trace;

	/* Main Loop */
	std::mutex mtx;
	std::thread th_h2d(loop_nbrs_h2d, std::ref(LRU), std::ref(window), slice_set, std::ref(nbrPosition), stream_h2d, std::ref(state_h2d_thread), std::ref(state_main_thread), std::ref(mtx));
	std::thread th_readdisk(loop_read_disk, std::ref(window), std::ref(prevPosX), std::ref(prevPosY), std::ref(curPosX), std::ref(curPosY), std::ref(light_field_size), std::ref(state_read_thread), std::ref(state_main_thread));

	while (1) {
#if 0 // AUTO MOVE
		if (dir % 3 == 0)
		{
			// curPosX--;  // DDZ
			// curPosY++;  // DDZ
			curPosX++;  // D
			// curPosY++;  // X 
			// curPosX++;  // WWD
		}
		else
		{
			// curPosX++;  // DDZ
			curPosX++;  // D
			// curPosY++;  // X
			// curPosY--;  // WWD
		}
		dir++;

		printf("\tPosition(%d, %d)\n", curPosX, curPosY);
		state_main_thread = MAIN_THREAD_WAIT;
		getNeighborList(nbrPosition, curPosX, curPosY);
		if (curPosY > 24) {
			break;
		}
#else
		prevPosX = curPosX;
		prevPosY = curPosY;

		if (state_main_thread != MAIN_THREAD_INIT) {
			if (getKey(curPosX, curPosY) < 0) {
				break;
			}
		}
		set_rendering_range(localPosX, localPosY, output_width_each_dir, curPosX, curPosY);

		state_main_thread = MAIN_THREAD_WAIT;
		getNeighborList(nbrPosition, curPosX, curPosY);
#endif
		sw.Start();
		state_main_thread = MAIN_THREAD_H2D;

		mtx.lock();
		std::pair<size_t, size_t> hitrate = cache_slice(LRU, window, slice_set, curPosX, curPosY);
		int mode = LRU.synchronize_HashmapOfPtr(window, stream_main, state_read_thread);
		mtx.unlock();

		state_main_thread = MAIN_THREAD_RENDERING;
		int twid = 2;
		int thei = 32;
		dim3 threadsPerBlock(twid, thei);
		// interlace mode -> block shape : 2250*1280

		dim3 blocksPerGrid_F((int)ceil((float)output_width_each_dir[0] / (float)twid), (int)ceil((float)(g_height / 2) / (float)thei)); // set a shape of the threads-per-block
		dim3 blocksPerGrid_R((int)ceil((float)output_width_each_dir[1] / (float)twid), (int)ceil((float)(g_height / 2) / (float)thei)); // set a shape of the threads-per-block
		dim3 blocksPerGrid_B((int)ceil((float)output_width_each_dir[2] / (float)twid), (int)ceil((float)(g_height / 2) / (float)thei)); // set a shape of the threads-per-block
		dim3 blocksPerGrid_L((int)ceil((float)output_width_each_dir[3] / (float)twid), (int)ceil((float)(g_height / 2) / (float)thei)); // set a shape of the threads-per-block

		rendering << < blocksPerGrid_F, threadsPerBlock, 0, stream_main >> > (u_synthesized_view, LRU.d_devPtr_hashmap_odd, LRU.d_devPtr_hashmap_even, 0, mode, 0, curPosX, curPosY, g_width, g_height, g_slice_width);
		cudaError_t err = cudaStreamSynchronize(stream_main);
		assert(err == cudaSuccess);
		rendering << < blocksPerGrid_R, threadsPerBlock, 0, stream_main >> > (u_synthesized_view, LRU.d_devPtr_hashmap_odd, LRU.d_devPtr_hashmap_even, output_width_each_dir[0], mode, 1, curPosX, curPosY, g_width, g_height, g_slice_width);
		err = cudaStreamSynchronize(stream_main);
		assert(err == cudaSuccess);
		rendering << < blocksPerGrid_B, threadsPerBlock, 0, stream_main >> > (u_synthesized_view, LRU.d_devPtr_hashmap_odd, LRU.d_devPtr_hashmap_even, output_width_each_dir[0] + output_width_each_dir[1], mode, 2, curPosX, curPosY, g_width, g_height, g_slice_width);
		err = cudaStreamSynchronize(stream_main);
		assert(err == cudaSuccess);
		rendering << < blocksPerGrid_L, threadsPerBlock, 0, stream_main >> > (u_synthesized_view, LRU.d_devPtr_hashmap_odd, LRU.d_devPtr_hashmap_even, output_width_each_dir[0] + output_width_each_dir[1] + output_width_each_dir[2],mode, 2, curPosX, curPosY, g_width, g_height, g_slice_width);
		err = cudaStreamSynchronize(stream_main);
		assert(err == cudaSuccess);

		state_main_thread = MAIN_THREAD_COMPLETE;

		double stop = sw.Stop();
		time_end_to_end.push_back(stop);
		reused_per_total.push_back(hitrate);
		field_mode.push_back(mode);
		position_trace.push_back(std::make_pair(curPosX, curPosY));
		int output_width = output_width_each_dir[0] + output_width_each_dir[1] + output_width_each_dir[2] + output_width_each_dir[3];
		printf("%dx%d image synthesized\n", output_width, g_height);
		printf("[%d] %f ms, Cached Slices: %d(Odd), %d(Even)\n", mode, stop, LRU.size(ODD), LRU.size(EVEN));

#if LOGGER==1
		FILE* fv = fopen(("./result/view/[" + std::to_string(output_width) + "x" + std::to_string(g_height) + "] " + IntToFormattedString(curPosX) + "_" + IntToFormattedString(curPosY) + ".bgr").c_str(), "wb");
		fwrite(u_synthesized_view, 1, g_output_width * g_height * 3, fv);
		fclose(fv);
#endif
	}
#if LOGGER==1
	FILE* fout_experimental_result = fopen(("./result/ours/" + IntToFormattedString(g_slice_width) + ".log").c_str(), "w");
	fprintf(fout_experimental_result, "mode\tposition\telapsed_time\tresued\ttotal\thitrate\n");
	for (int i = 0; i < time_end_to_end.size(); i++)
	{
		fprintf(fout_experimental_result, "%d\t%d,%d\t%f\t%d\t%d\t%f\n", field_mode.at(i), position_trace.at(i).first, position_trace.at(i).second, time_end_to_end.at(i), reused_per_total.at(i).first, reused_per_total.at(i).second, (double)reused_per_total.at(i).first / (double)reused_per_total.at(i).second);
	}
	fclose(fout_experimental_result);
#endif
	state_main_thread = MAIN_THREAD_TERMINATED;

	/* Destruct */
	if (th_h2d.joinable())
	{
		state_h2d_thread = H2D_THREAD_TERMINATED;
		th_h2d.join();
	}
	if (th_readdisk.joinable())
	{
		state_read_thread = READ_DISK_THREAD_TERMINATED;
		th_readdisk.join();
	}

	free_uint8(u_synthesized_view, "unified");
	cudaStreamDestroy(stream_main);
	cudaStreamDestroy(stream_h2d);

	return 0;
}