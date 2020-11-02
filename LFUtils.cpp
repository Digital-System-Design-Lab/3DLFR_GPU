#include "LFUtils.h"

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
	case 'w': {			posY--; } break;
	case 'e': {	posX++;	posY--; } break;
	case 'd': {	posX++;			} break;
	case 'c': {	posX++;	posY++;	} break;
	case 'x': {			posY++; } break;
	case 'z': {	posX--; posY++; } break;
	case 'a': {	posX--; } break;
	case 'q': {	posX--; posY--; } break;
	case 27: {	printf("Terminate\n"); return -1;	}
	default: break;
	}
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


double differentiation(double prev, double cur, double timespan)
{
	return (cur - prev) / timespan;
}

std::pair<double, double> deadReckoning(std::pair<double, double> a_2, std::pair<double, double> a_1, std::pair<double, double> a0, double framerate, int f)
{
	double tf = (double)f / framerate;

	std::pair<double, double> v0;
	std::pair<double, double> v_1;
	std::pair<double, double> c0;

	v0 = std::make_pair(differentiation(a_1.first, a0.first, 1.0 / framerate), differentiation(a_1.second, a0.second, 1.0 / framerate));
	v_1 = std::make_pair(differentiation(a_2.first, a_1.first, 1.0 / framerate), differentiation(a_2.second, a_1.second, 1.0 / framerate));
	c0 = std::make_pair(differentiation(v_1.first, v0.first, tf), differentiation(v_1.second, v0.second, tf));

	return std::make_pair(a0.first + v0.first * tf + 0.5 * c0.first * tf * tf, a0.second + v0.second * tf + 0.5 * c0.second * tf * tf);;
}

std::vector<std::pair<double, std::pair<int, int>>> doDeadReckoning(double prevprevPosX, double prevprevPosY, double prevPosX, double prevPosY, double curPosX, double curPosY, double framerate, int pred)
{
	std::pair<double, double> prevprevPos = std::make_pair(prevprevPosX, prevprevPosY);
	std::pair<double, double> prevPos = std::make_pair(prevPosX, prevPosY);
	std::pair<double, double> curPos = std::make_pair(curPosX, curPosY);

	std::vector<std::pair<double, std::pair<int, int>>> v;

	for (int i = 1; i <= pred; i++)
	{
		std::pair<double, double> nextPos = deadReckoning(prevprevPos, prevPos, curPos, framerate, i);
		nextPos.second = (double)clamp(nextPos.second, 1, 25);
		v.push_back(std::make_pair(1.0 / pow(2,i), std::make_pair((int)nextPos.first, (int)nextPos.second)));
		// printf("after-%d frame : %f\n", i, deadReckoning(prevprevPosX, prevPosX, curPosX, 90.0, i));
	}

	return v;
}

Interlaced_LF* get_LF_from_Window(std::vector<Interlaced_LF>& window, const int& LF_number)
{
	while (1) {
		for (std::vector<Interlaced_LF>::iterator iter = window.begin(); iter != window.end(); iter++) {
			if (LF_number == iter->LF_number) {
				return &*iter;
			}
		}
		// printf("updating window now...\n");
	}
}