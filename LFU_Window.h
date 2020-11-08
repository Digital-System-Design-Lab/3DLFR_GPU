#ifndef LFU_WINDOW_H_
#define LFU_WINDOW_H_

#include "LF_Utils.cuh"
extern std::string BMW_LF[30][4];

struct LFU {
	int id;
	Interlaced_LF* LF[4] = { nullptr, nullptr, nullptr, nullptr };
	LFU* nbr[8] = { nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
};

class LFU_Window {
public:
	LFU_Window(const int& posX, const int& posY, const int& light_field_size);
	~LFU_Window();
	void update_window(const int& prevPosX, const int& prevPosY, const int& curPosX, const int& curPosY, const int& light_field_size, const MAIN_THREAD_STATE& main_thread_state);

	LFU* m_center;
	uint8_t* m_pinnedLFU[2][4];
	PINNED_LFU_STATUS pinned_memory_status;
private:
	void construct_window(const int& light_field_size);

	Interlaced_LF m_row[12];
	Interlaced_LF m_col[12];
	LFU m_LFU[9];
};

#endif