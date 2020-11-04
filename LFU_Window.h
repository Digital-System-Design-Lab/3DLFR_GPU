#ifndef LFU_WINDOW_H_
#define LFU_WINDOW_H_

#include "LFUtils.cuh"
extern std::string BMW_LF[30][4];

struct LFU {
	Interlaced_LF* front;
	Interlaced_LF* right;
	Interlaced_LF* back;
	Interlaced_LF* left;

	LFU* N = nullptr;
	LFU* NE = nullptr;
	LFU* E = nullptr;
	LFU* SE = nullptr;
	LFU* S = nullptr;
	LFU* SW = nullptr;
	LFU* W = nullptr;
	LFU* NW = nullptr;
};

class LFU_Window {
public:
	LFU_Window(const int& posX, const int& posY, const int& light_field_size);
	~LFU_Window();
	void update_window();
private:
	void construct_window(const int& light_field_size);

	Interlaced_LF m_row[12];
	Interlaced_LF m_col[12];
	LFU m_LFU[9];
	LFU* m_center;
};

#endif