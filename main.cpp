#include "LF_Renderer.cuh"

int main()
{
	std::string PixelRange = "S:/PixelRange/";
	std::string LF = "S:/BMW_4K/";
	LF_Renderer renderer(PixelRange, LF);

	int curPosX = 150;
	int curPosY = 150;

	while (1)
	{
		if(getKey(curPosX, curPosY) < 0) break;
		uint8_t* synthesized_view = renderer.do_rendering(curPosX, curPosY);

		FILE* fv = fopen(("./result/view/[" + std::to_string(9000) + "x" + std::to_string(2048) + "] " + IntToFormattedString(curPosX) + "_" + IntToFormattedString(curPosY) + ".bgr").c_str(), "wb");
		fwrite(synthesized_view, 1, 4096 * 2048 * 3, fv);
		fclose(fv);
	}
	renderer.terminate();

	return 0;
}