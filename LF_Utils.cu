#include "LF_Utils.cuh"

uint8_t* alloc_uint8(int size, std::string alloc_type) {
	uint8_t* buf;
	if (alloc_type == "pinned") {
		cudaMallocHost((void**)&buf, size);
		memset(buf, 0, size);
	}
	else if (alloc_type == "pageable") {
		buf = new uint8_t[size]();
		memset(buf, 0, size);

	}
	else if (alloc_type == "device") {
		cudaMalloc((void**)&buf, size);
		cudaMemset(buf, 0, size);
	}
	else if (alloc_type == "unified") {
		cudaMallocManaged((void**)&buf, size);
	}
	else exit(1);

	return buf;
}

void free_uint8(uint8_t* buf, std::string alloc_type) {
	if (alloc_type == "pinned") {
		cudaFreeHost(buf);
	}
	else if (alloc_type == "pageable") {
		delete[] buf;
	}
	else if (alloc_type == "device" || alloc_type == "unified") {
		cudaFree(buf);
	}
	else exit(1);
}

int read_uint8(uint8_t* buf, std::string filename, int size)
{
	int fd;
	int ret;

	fd = open(filename.c_str(), O_RDONLY | O_BINARY);
	ret = fd;
	if (ret < 0) {
		printf("open failed, %s\n", filename.c_str());
		assert(ret == 0);
		exit(1);
	}

	if (size < 0) {
		if ((ret = lseek(fd, 0, SEEK_END)) < 0) {
			printf("SEEK_END failed, %s\n", filename.c_str());
			assert(ret == 0);
			exit(1);
		}
		if ((ret = tell(fd)) < 0) {
			printf("tell failed, %s\n", filename.c_str());
			assert(ret == 0);
			exit(1);
		}
		size = ret;
		if ((ret = lseek(fd, 0, SEEK_SET)) < 0) {
			printf("SEEK_SET failed, %s\n", filename.c_str());
			assert(ret == 0);
			exit(1);
		}
	}

	ret = read(fd, buf, sizeof(uint8_t) * size); // x64
	close(fd);

	if (ret != size) {
		printf("read failed, %s\n", filename.c_str());
		assert(ret == size);
		exit(1);
	}

	return ret;
}

int write_uint8(uint8_t* buf, std::string filename, int size)
{
	int fd;
	if ((fd = open(filename.c_str(), O_WRONLY | O_BINARY)) < 0) return fd;
	if (size < 0) size = _msize(buf);

	int ret = write(fd, buf, sizeof(uint8_t) * size); // x64 
	close(fd);

	return ret;
}

void StopWatch::Start() {
	t0 = std::chrono::high_resolution_clock::now();
}

double StopWatch::Stop() {
	double stop = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch() - t0.time_since_epoch()).count();
	return stop / 1000000.0; // ms
}

double getEuclideanDist(int x, int y, int origX, int origY)
{
	return sqrt(pow(((double)x - (double)origX), 2) + pow(((double)y - (double)origY), 2));
}

int clamp(int val, int min, int max)
{
	if (val > max)
		return max;
	else if (val < min)
		return min;
	else return val;
}

double rad2deg(double rad)
{
	return (rad * 180.0 / PI);
}

double deg2rad(double deg)
{
	return (deg * PI / 180.0);
}

float deg2rad(float deg)
{
	return (deg * PI / 180.0f);
}

void minmax(int val, int& min, int& max)
{
	min = (val < min) ? val : min;
	max = (val > max) ? val : max;
}

int getKey(int& posX, int& posY)
{
	printf("%d, %d -> ", posX, posY);
	int c = getch();

	switch (c)
	{
	case 'x': {			posY--; } break;
	case 'c': {	posX++;	posY--; } break;
	case 'd': {	posX++;			} break;
	case 'e': {	posX++;	posY++;	} break;
	case 'w': {			posY++; } break;
	case 'q': {	posX--; posY++; } break;
	case 'a': {	posX--; } break;
	case 'z': {	posX--; posY--; } break;
	case 27: {	printf("Terminate\n"); return -1;	}
	default: break;
	}

	posX = clamp(posX, 101, 499);
	posY = clamp(posY, 101, 5499);

	printf("%d, %d\n", posX, posY);

	return 0;
}

std::string IntToFormattedString(int n)
{
	if (n < 10)
		return ("000" + std::to_string(n));
	else if (n < 100)
		return ("00" + std::to_string(n));
	else if (n < 1000)
		return ("0" + std::to_string(n));
	else
		return (std::to_string(n));
}

std::string FloatToFormattedString(float f)
{
	if (f < 10)
		return ("000" + std::to_string(f));
	else if (f < 100)
		return ("00" + std::to_string(f));
	else if (f < 1000)
		return ("0" + std::to_string(f));
	else
		return (std::to_string(f));
}

int preRendering(int x, int z, int dir)
{	
	float fov = 90.0f;
	float times = 270.f;

	int errCode = 0;

	int DATAW = 50;
	int LFUW = 100;
	int Y = LFUW / 2;

	int localPosX, localPosZ;

	localPosX = x % 100 - 50;
	localPosZ = z % 100 - 50;
	if (dir == 1) {
		localPosX = -1 * localPosZ;
		localPosZ = localPosX;
	}
	else if (dir == 2) {
		localPosX = -1 * localPosX;
		localPosZ = -1 * localPosZ;
	}
	else if (dir == 3) {
		localPosX = localPosZ;
		localPosZ = -1 * localPosX;
	}

	float theta_L = rad2deg(atan2((-1.0 * LFUW / 2 - localPosX), (LFUW / 2 - localPosZ)));
	float theta_R = rad2deg(atan2((1.0 * LFUW / 2 - localPosX), (LFUW / 2 - localPosZ)));

	int prevP = 1e6;
	int min = 1e6;
	int max = -1e6;

	std::string filename = "S:/4K/LFU/" + std::to_string(x) + "_" + std::to_string(z) + ".txt";
	FILE* fp = fopen(filename.c_str(), "a");
	int output_width = (theta_R - theta_L) / 0.04f;
	// printf("%f - (%f) = %f -> %d\n",theta_R, theta_L, output_width);
	for (int w = 0; w < output_width; w++)
	{
		float theta_P = theta_L + (0.04f * (float)w); // 가져올 ray가 front와 이루는 각 (rad)
		// float xP = x0 + z0 * tanf(deg2rad(theta_P)); // tan -> 구간 내에서 odd function
		float xP = (float)(Y - localPosZ) * tanf(deg2rad(theta_P)) + localPosX; // tan -> 구간 내에서 odd function
		float b = sqrtf(2.0f) * LFUW;
		float N_dist = sqrt((float)((xP - localPosX) * (xP - localPosX) + (Y - localPosZ) * (Y - localPosZ))) / b;

		xP /= 2;
		int P_1 = (int)round(xP + (DATAW / 2));
		if (dir == 2 || dir == 3) P_1 = DATAW - P_1 - 1;
		P_1 = clamp(P_1, 0, DATAW - 1);
		// P_1 = Clamp(P_1, 0, LEN_CAM_ARRAY - 1);

		float U = (theta_P * ((1.0) / (180.0))) * WIDTH / 2 + WIDTH / 2;
		int U_1 = (int)(roundf(U));

		if (dir == 1) U_1 += WIDTH / 4;
		if (dir == 2) U_1 += WIDTH / 2;
		if (dir == 3) U_1 -= WIDTH / 4;
		U_1 %= WIDTH;
		U_1 = clamp(U_1, 0, WIDTH - 1);
		if ((prevP != 1e6 && prevP != P_1) || w == output_width - 1)
		{
			fprintf(fp, "%d\t%d\t%d\t%d\n", dir, prevP, min, max);
			min = 1e6;
			max = -1e6;
		}
		prevP = P_1;
		minmax(U_1, min, max);

	}

	fclose(fp);
	return 0;
}

void write_rendering_range()
{
	for (int dir = 0; dir < 4; dir++) {
		for (int y = 0; y < 100; y++) {
			for (int x = 0; x < 100; x++) {
				preRendering(x, y, dir);
			}
		}
	}
}

int getLFUID(const int& posX, const int& posY)
{
	return 56 * (posX / 100) + (posY / 100);
}

void find_LF_number_BMW(int& front, int& right, int& back, int& left, const int& LFUID)
{ // 56x6 LFU
	left = LFUID;
	right = left + 56;
	back = (LFUID / 56) + (6 * (LFUID % 56)) + 392;
	front = back + 6;
}

void getLocalPosition(int& localPosX, int& localPosY, const int& curPosX, const int& curPosY)
{
	localPosX = curPosX % 100 - 50;
	localPosY = curPosY % 100 - 50;
}

void write_bmw_fname_array(std::string path) {
	FILE* fp = fopen(path.c_str(), "w");
	fprintf(fp, "std::string BMW_LF[392][4] = \n");
	fprintf(fp, "{\n");
	for (int i = 0; i < 392; i++) {
		fprintf(fp, "\t{\n");

		int f, r, b, l;
		find_LF_number_BMW(f, r, b, l, i);
		fprintf(fp, "\t\t\"Row%d\",\n", f);
		fprintf(fp, "\t\t\"Column%d\",\n", r);
		fprintf(fp, "\t\t\"Row%d\",\n", b);
		fprintf(fp, "\t\t\"Column%d\"\n", l);
		if (i == 391) fprintf(fp, "\t}\n");
		else fprintf(fp, "\t},\n");
	}
	fprintf(fp, "};");
	fclose(fp);
}

void constructLF_interlace() { // BMW LF configuration.xlsx 참고
/*
	unsigned char* LF_odd = new unsigned char[HEIGTH*WIDTH / 2 * 3];
	unsigned char* LF_even = new unsigned char[HEIGTH*WIDTH / 2 * 3];
	char OUT_FILE_odd[128];
	char OUT_FILE_even[128];
	char LF_FILE[128];
	//char LF_FILE[100] = "LFU360/191020/Asite/Row10/0001.jpg";
	int N = 50;
	FILE* fp_odd;
	FILE* fp_even;

	for (int col = 1; col <= 392; col++)
	{
		//sprintf_s(OUT_FILE, sizeof(OUT_FILE), "LFU360/191025_output/Row%d.LF", count);
		sprintf_s(OUT_FILE_odd, sizeof(OUT_FILE_odd), "E:/BMW_4K/Column%d_odd.bgr", col - 1);
		sprintf_s(OUT_FILE_even, sizeof(OUT_FILE_even), "E:/BMW_4K/Column%d_even.bgr", col - 1);
		fopen_s(&fp_odd, OUT_FILE_odd, "wb");
		fopen_s(&fp_even, OUT_FILE_even, "wb");
		for (int no = 1; no <= N; no++) {
			if (no < 10)
			{
				sprintf_s(LF_FILE, sizeof(LF_FILE), "F:/BMW/Column%d/000%d.jpg", col, no);
			}
			else if (no < 100)
			{
				sprintf_s(LF_FILE, sizeof(LF_FILE), "F:/BMW/Column%d/00%d.jpg", col, no);
			}
			else
			{
				sprintf_s(LF_FILE, sizeof(LF_FILE), "F:/BMW/Column%d/0%d.jpg", col, no);
			}
			cv::Mat src_img = cv::imread(LF_FILE);
			cv::Mat img;
			resize(src_img, img, Size(TARGET_W, TARGET_H));

			for (int w = 0; w < TARGET_W; w++) {
				for (int h = 0; h < TARGET_H / 2; h++) {
					LF_odd[w * TARGET_H / 2 * 3 + h * 3 + 0] = img.at<Vec3b>(2 * h, w)[0];
					LF_odd[w * TARGET_H / 2 * 3 + h * 3 + 1] = img.at<Vec3b>(2 * h, w)[1];
					LF_odd[w * TARGET_H / 2 * 3 + h * 3 + 2] = img.at<Vec3b>(2 * h, w)[2];

					LF_even[w * TARGET_H / 2 * 3 + h * 3 + 0] = img.at<Vec3b>(2 * h + 1, w)[0];
					LF_even[w * TARGET_H / 2 * 3 + h * 3 + 1] = img.at<Vec3b>(2 * h + 1, w)[1];
					LF_even[w * TARGET_H / 2 * 3 + h * 3 + 2] = img.at<Vec3b>(2 * h + 1, w)[2];
				}
			}
			fwrite(LF_odd, 1, TARGET_W* TARGET_H / 2 * 3, fp_odd);
			fwrite(LF_even, 1, TARGET_W* TARGET_H / 2 * 3, fp_even);
		}
		printf("Read col %d IMAGE..\n", col);
		fclose(fp_odd);
		fclose(fp_even);
	}

	for (int row = 1; row <= 342; row++)
	{
		int newRowNum = 336 - (6 * ((row - 1) / 6)) + ((row - 1) % 6) + 392;
		//sprintf_s(OUT_FILE, sizeof(OUT_FILE), "LFU360/191025_output/Row%d.LF", count);
		sprintf_s(OUT_FILE_odd, sizeof(OUT_FILE_odd), "E:/BMW_4K/Row%d_odd.bgr", newRowNum);
		sprintf_s(OUT_FILE_even, sizeof(OUT_FILE_even), "E:/BMW_4K/Row%d_even.bgr", newRowNum);
		fopen_s(&fp_odd, OUT_FILE_odd, "wb");
		fopen_s(&fp_even, OUT_FILE_even, "wb");
		for (int no = 1; no <= N; no++) {
			if (no < 10)
			{
				sprintf_s(LF_FILE, sizeof(LF_FILE), "F:/BMW/Row%d/000%d.jpg", row, no);
			}
			else if (no < 100)
			{
				sprintf_s(LF_FILE, sizeof(LF_FILE), "F:/BMW/Row%d/00%d.jpg", row, no);
			}
			else
			{
				sprintf_s(LF_FILE, sizeof(LF_FILE), "F:/BMW/Row%d/0%d.jpg", row, no);
			}
			cv::Mat src_img = cv::imread(LF_FILE);
			cv::Mat img;
			resize(src_img, img, Size(TARGET_W, TARGET_H));

			for (int w = 0; w < TARGET_W; w++) {
				for (int h = 0; h < TARGET_H / 2; h++) {
					LF_odd[w * TARGET_H / 2 * 3 + h * 3 + 0] = img.at<Vec3b>(2 * h, w)[0];
					LF_odd[w * TARGET_H / 2 * 3 + h * 3 + 1] = img.at<Vec3b>(2 * h, w)[1];
					LF_odd[w * TARGET_H / 2 * 3 + h * 3 + 2] = img.at<Vec3b>(2 * h, w)[2];

					LF_even[w * TARGET_H / 2 * 3 + h * 3 + 0] = img.at<Vec3b>(2 * h + 1, w)[0];
					LF_even[w * TARGET_H / 2 * 3 + h * 3 + 1] = img.at<Vec3b>(2 * h + 1, w)[1];
					LF_even[w * TARGET_H / 2 * 3 + h * 3 + 2] = img.at<Vec3b>(2 * h + 1, w)[2];
				}
			}
			fwrite(LF_odd, 1, TARGET_W* TARGET_H / 2 * 3, fp_odd);
			fwrite(LF_even, 1, TARGET_W* TARGET_H / 2 * 3, fp_even);
		}
		printf("Read row %d IMAGE..\n", newRowNum);
		fclose(fp_odd);
		fclose(fp_even);
	}
*/
}

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

__device__ int dev_getLFUID(const int& posX, const int& posY)
{
	return 56 * (posX / 100) + (posY / 100);
}

__device__ int dev_find_LF_number_BMW(const int& direction, const int& posX, const int& posY)
{
	int LFUID = dev_getLFUID(posX, posY);

	switch (direction)
	{
	case 0: return (LFUID / 56) + (6 * (LFUID % 56)) + 392 + 6;
		break;
	case 1: return LFUID + 56;
		break;
	case 2: return (LFUID / 56) + (6 * (LFUID % 56)) + 392;
		break;
	case 3: return LFUID; 
		break;
	default: return -1;
		break;
	}
}