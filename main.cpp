#include "LF_Renderer.cuh"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

int getKeywithCV(int& posX, int& posY)
{
	printf("%d, %d -> ", posX, posY);
	int c = cvWaitKey(0);

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

int main()
{
	int curPosX = 150;
	int curPosY = 150;

	std::string PixelRange = "S:/PixelRange/";
	std::string LF = "S:/BMW_4K/";

	LF_Renderer renderer(PixelRange, LF, curPosX, curPosY);

	cv::Mat img = cv::Mat(2048, 9000, CV_8UC3);
	cv::namedWindow("window", CV_WINDOW_NORMAL);
	cv::resizeWindow("window", 2250, 512);
	while (1)
	{
		uint8_t* synthesized_view = renderer.do_rendering(curPosX, curPosY);
		img.data = synthesized_view;
		cv::imshow("window", img);
		if(getKeywithCV(curPosX, curPosY) < 0) break;
	}
	renderer.terminate();

	return 0;
}