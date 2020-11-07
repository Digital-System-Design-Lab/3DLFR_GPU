#ifndef ENUMS_H_
#define ENUMS_H_

enum MAIN_THREAD_STATE {
	MAIN_THREAD_TERMINATED = -3,
	MAIN_THREAD_INIT = -2,
	MAIN_THREAD_WAIT = -1,
	MAIN_THREAD_H2D = 0,
	MAIN_THREAD_RENDERING = 1,
	MAIN_THREAD_D2H = 2,
	MAIN_THREAD_COMPLETE = 3
};

enum H2D_THREAD_STATE {
	H2D_THREAD_TERMINATED = -4,
	H2D_THREAD_INIT = -3,
	H2D_THREAD_INTERRUPTED = -2,
	H2D_THREAD_WAIT = -1,
	H2D_THREAD_RUNNING = 0
};

enum READ_DISK_THREAD_STATE {
	READ_DISK_THREAD_TERMINATED = -2,
	READ_DISK_THREAD_INIT = -1,
	READ_DISK_THREAD_CURRENT_LF_READING = 0,
	READ_DISK_THREAD_CURRENT_LF_READ_COMPLETE = 1,
	READ_DISK_THREAD_NEIGHBOR_LF_READING = 2,
	READ_DISK_THREAD_NEIGHBOR_LF_READ_COMPLETE = 3,
};

enum LF_READ_PROGRESS {
	LF_READ_PROGRESS_NOT_PREPARED = -1,
	LF_READ_PROGRESS_ODD_FIELD_PREPARED = 0,
	LF_READ_PROGRESS_EVEN_FIELD_PREPARED = 1
};

enum ROW_COL {
	ROW = 0,
	COL = 1
};

enum FOUR_DIRECTION {
	FRONT = 0,
	RIGHT = 1,
	BACK = 2,
	LEFT = 3
};

enum INTERLACE_FIELD {
	ODD = 0,
	EVEN = 1
};

enum PINNED_LFU_STATUS {
	PINNED_LFU_NOT_AVAILABLE = -1,
	PINNED_LFU_ODD_AVAILABLE = 0,
	PINNED_LFU_EVEN_AVAILABLE = 1
};

enum LFU_NEIGHBOR {
	N = 0,
	NE = 1,
	E = 2,
	SE = 3,
	S = 4,
	SW = 5,
	W = 6,
	NW = 7
};

#endif