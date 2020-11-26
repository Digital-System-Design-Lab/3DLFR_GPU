#ifndef LFU_WINDOW_H_
#define LFU_WINDOW_H_

#include "LF_Utils.cuh"
extern std::string BMW_LF[392][4];

#define LFU_WIDTH 100

struct LFU {
	int id;
	Interlaced_LF* LF[4] = { nullptr, nullptr, nullptr, nullptr };
	LFU* nbr[8] = { nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
};

class LFU_Window {
public:
	LFU_Window(const int& posX, const int& posY, const size_t& light_field_size, const std::string& dir);
	~LFU_Window();
	int update_window(const int& prevPosX, const int& prevPosY, const int& curPosX, const int& curPosY, const size_t& light_field_size, const MAIN_THREAD_STATE& main_thread_state);

	LFU* m_center;
	uint8_t* m_pinnedLFU[2][4];
	PINNED_LFU_STATUS pinned_memory_status;
private:

	
	int read_uint8(uint8_t* buf, std::string filename, const INTERLACE_FIELD& field, int size = -1);
	void construct_window(const size_t& light_field_size);

	Interlaced_LF m_row[12];
	Interlaced_LF m_col[12];
	LFU m_LFU[9];

	std::string LF_prefix;
};

#endif