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

__global__ void rendering(uint8_t* outImage, uint8_t** d_hashmap_odd, uint8_t** d_hashmap_even, int mode, int posX, int posY, int g_width, int g_height, int g_slice_width, float fov = 90.0f, float times = 270.0f)
{
	int tw = blockIdx.x * blockDim.x + threadIdx.x; // blockIdx.x = (int)[0, (out_w - 1)]
	int th = blockIdx.y * blockDim.y + threadIdx.y; // threadIdx = (int)[0, (g_height - 1)]

	int LFUW = 100;
	int z0 = posY; 
	int x0 = posX; 

	float theta_L = -fov / 2.0;
	float theta_R = fov / 2.0;

	float theta_P = theta_L + (0.04 * (float)tw);
	int Y = LFUW / 2;
	float b = sqrt(2.0) * LFUW;
	float xP = x0 + z0 * __tanf(dev_deg2rad(theta_P)); 

	float N_dist = sqrt((float)((xP - x0) * (xP - x0) + (Y - z0) * (Y - z0))) / b; 
	int P_1 = (int)(roundf(xP));
	// float U = (theta_P / (fov / 2.0)) * WIDTH / 2 + WIDTH / 2;
	float U = (theta_P * (1.0f / 180.0f)) * WIDTH / 2 + WIDTH / 2;

	int U_1 = (int)(roundf(U));
	int U_1_n = 0;
	int N_off = (int)(roundf(times * N_dist + 0.5)) >> 1;
	
	U_1 %= WIDTH;
	U_1 = dev_Clamp(U_1, 0, WIDTH - 1);

	int LF_num = P_1 / LENGTH;
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
	uint8_t oddpel_ch0 = d_hashmap_odd[slice][(pixel_col * g_height / 2) * 3 + H_1 * 3 + 0]; // Random access to pixel column
	uint8_t oddpel_ch1 = d_hashmap_odd[slice][(pixel_col * g_height / 2) * 3 + H_1 * 3 + 1]; // Random access to pixel column
	uint8_t oddpel_ch2 = d_hashmap_odd[slice][(pixel_col * g_height / 2) * 3 + H_1 * 3 + 2]; // Random access to pixel column
	outImage[(2 * th) * (OUTPUT_WIDTH * 3) + tw * 3 + 0] = oddpel_ch0; // b 
	outImage[(2 * th) * (OUTPUT_WIDTH * 3) + tw * 3 + 1] = oddpel_ch1; // g 
	outImage[(2 * th) * (OUTPUT_WIDTH * 3) + tw * 3 + 2] = oddpel_ch2; // r 

	if (mode == 1) {
		uint8_t evenpel_ch0 = d_hashmap_even[slice][(pixel_col * g_height / 2) * 3 + H_1 * 3 + 0]; // Random access to pixel column
		uint8_t evenpel_ch1 = d_hashmap_even[slice][(pixel_col * g_height / 2) * 3 + H_1 * 3 + 1]; // Random access to pixel column
		uint8_t evenpel_ch2 = d_hashmap_even[slice][(pixel_col * g_height / 2) * 3 + H_1 * 3 + 2]; // Random access to pixel column

		outImage[(2 * th + 1) * (OUTPUT_WIDTH * 3) + tw * 3 + 0] = evenpel_ch0; // b 
		outImage[(2 * th + 1) * (OUTPUT_WIDTH * 3) + tw * 3 + 1] = evenpel_ch1; // g 
		outImage[(2 * th + 1) * (OUTPUT_WIDTH * 3) + tw * 3 + 2] = evenpel_ch2; // r 
	}
	else
	{
		outImage[(2 * th + 1) * (OUTPUT_WIDTH * 3) + tw * 3 + 0] = oddpel_ch0; // b 
		outImage[(2 * th + 1) * (OUTPUT_WIDTH * 3) + tw * 3 + 1] = oddpel_ch1; // g 
		outImage[(2 * th + 1) * (OUTPUT_WIDTH * 3) + tw * 3 + 2] = oddpel_ch2; // r 
	}

}

void load_slice_set(SliceSet slice_set[][100]) {
	for (int x = 0; x < 100; x++) {
		for (int y = 0; y < 100; y++)
		{
			std::string fname = "S:/LFU/" + std::to_string(x) + "_" + std::to_string(y) + ".txt";
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

	for (SliceSet::iterator it = slice_set[posX][posY].begin(); it != slice_set[posX][posY].end; it++)
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

int cache_slice_in_background(LRUCache& LRU, uint8_t* pinnedLFU[2][4], std::vector<std::pair<int, int>>& nbrPosition, cudaStream_t stream_h2d, H2D_THREAD_STATE& thread_state_h2d, const MAIN_THREAD_STATE& thread_state_main) {

/*
	int i = 0;
	int s = 0;
	while (1)
	{
		while (1) {
			for (int p = 0; p < 8; p++) {
				if (thread_state_main < MAIN_THREAD_RENDERING) {
					thread_state_h2d = H2D_THREAD_INTERRUPTED;
					return -1;
				} // interrupted

				int posX_at_p = nbrPosition.at(p).first;
				int posY_at_p = nbrPosition.at(p).second;

				int leftend_image, rightend_image;
				set_both_end_image(leftend_image, rightend_image, posX_at_p, posY_at_p);

				int img = leftend_image + i;
				std::pair<int, int> slice_range = slice_map[posY_at_p].at(i); 
				int slice_num = slice_range.first + s; 

				if (i < slice_map[posY_at_p].size() && slice_num <= slice_range.second) 
				{
					SliceID id;

					id.lf_number = img / g_length;
					id.image_number = img % g_length;
					id.slice_number = slice_num;

					int slice_location = find_slice_from_LF(id.image_number, id.slice_number, true);
					Interlaced_LF* LF = get_LF_from_Window(window, id.lf_number);

					if (LF->progress == LF_READ_PROGRESS_ODD_FIELD_PREPARED) {
						LRU.put(id, LF->odd_field + slice_location, stream_h2d, thread_state_h2d, ODD);
					}
					if (LF->progress == LF_READ_PROGRESS_EVEN_FIELD_PREPARED) {
						LRU.put(id, LF->odd_field + slice_location, stream_h2d, thread_state_h2d, ODD);
						LRU.put(id, LF->even_field + slice_location, stream_h2d, thread_state_h2d, EVEN);
					}
				}
			}
			i++;
			if (i >= slice_map[nbrPosition.back().second].size()) {
				i = 0;
				break;
			}
		}
		s++;
		if (s > slice_map[nbrPosition.back().second].back().second) return 0;
	}
*/
}

void loop_nbrs_h2d(LRUCache& LRU, uint8_t* pinnedLFU[2][4], std::vector<std::pair<int, int>>& nbrPosition, cudaStream_t stream_h2d, H2D_THREAD_STATE& thread_state_h2d, const MAIN_THREAD_STATE& thread_state_main, std::mutex& mtx)
{
	bool loop = true;
	while (loop) {
		mtx.lock();
		// cache_slice_in_background(LRU, pinnedLFU, nbrPosition, stream_h2d, thread_state_h2d, thread_state_main);
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

int main()
{
	/* Declare */
	StopWatch sw; // for benchmark
	const int limit_cached_slice = 500;
	const int limit_hashing_LF = 71;

	printf("Input resolution : %dx%dx%d\n", g_width, g_height, g_length);
	printf("Output resolution : %dx%d\n", g_output_width, g_height);
	printf("Slice resolution : %dx%d\n", g_slice_width, g_height);
	printf("Slice Cache Size Limit : %f MB\n", g_slice_size * limit_cached_slice / 1e6 * 2);
	printf("Hashing LF Range Limit : %d to %d\n", 0, limit_hashing_LF);

	const int light_field_size = g_width * g_height *g_length * 3;

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
	LFU_Window window(curPosX, curPosY, light_field_size / 2);
	uint8_t* u_synthesized_view = alloc_uint8(g_output_width * g_height * 3, "unified");

	int twid = 2;
	int thei = 32;
	dim3 threadsPerBlock(twid, thei); 
	// interlace mode -> block shape : 2250*1280
	dim3 blocksPerGrid((int)ceil((float)g_output_width / (float)twid), (int)ceil((float)(g_height / 2) / (float)thei)); // set a shape of the threads-per-block

	MAIN_THREAD_STATE state_main_thread;
	H2D_THREAD_STATE state_h2d_thread;
	READ_DISK_THREAD_STATE state_read_thread;

	state_main_thread = MAIN_THREAD_INIT;
	state_h2d_thread = H2D_THREAD_INIT;
	state_read_thread = READ_DISK_THREAD_NEIGHBOR_LF_READ_COMPLETE;
	 
	int dir = 0;
	int while_iter = 0;

	// for result analysis
	std::vector<double> time_end_to_end;
	std::vector<std::pair<size_t, size_t>> reused_per_total;
	std::vector<int> field_mode;
	std::vector<std::pair<int, int>> position_trace;

	/* Main Loop */
	std::mutex mtx;
	std::thread th_h2d(loop_nbrs_h2d, std::ref(LRU), std::ref(window.m_pinnedLFU), std::ref(nbrPosition), stream_h2d, std::ref(state_h2d_thread), std::ref(state_main_thread), std::ref(mtx));
	std::thread th_readdisk(loop_read_disk, std::ref(window), std::ref(prevPosX), std::ref(prevPosY), std::ref(curPosX), std::ref(curPosY), std::ref(state_read_thread), std::ref(state_main_thread));

	while (while_iter < 195) {
		while_iter++;
		prevPosX = curPosX;
		prevPosY = curPosY;

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
		if (state_main_thread != MAIN_THREAD_INIT) {
			if (getKey(curPosX, curPosY) < 0) {
				break;
			}
		}

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
		rendering << < blocksPerGrid, threadsPerBlock, 0, stream_main >> > (u_synthesized_view, LRU.d_devPtr_hashmap_odd, LRU.d_devPtr_hashmap_even, mode, curPosX, curPosY, g_width, g_height, g_slice_width);
		cudaStreamSynchronize(stream_main);
		// main_thread_state = MAIN_THREAD_D2H;
		// cudaMemcpyAsync(synthesized_view, u_synthesized_view, g_output_width * g_height * 3, cudaMemcpyDeviceToHost, stream_main); 
		state_main_thread = MAIN_THREAD_COMPLETE;

		double stop = sw.Stop();
		time_end_to_end.push_back(stop);
		reused_per_total.push_back(hitrate);
		field_mode.push_back(mode);
		position_trace.push_back(std::make_pair(curPosX, curPosY));
		printf("[%d] %f ms, Cached Slices: %d(Odd), %d(Even)\n", mode, stop, LRU.size(ODD), LRU.size(EVEN));

#if LOGGER==1
		FILE* fv = fopen(("./result/view/[" + std::to_string(g_output_width) + "x" + std::to_string(g_height) + "] " + IntToFormattedString(curPosX) + "_" + IntToFormattedString(curPosY) + ".bgr").c_str(), "wb");
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