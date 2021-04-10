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

enum DISK_READ_THREAD_STATE {
	DISK_READ_THREAD_TERMINATED = -2,
	DISK_READ_THREAD_CENTER_LFU_READING = -1,
	DISK_READ_THREAD_CENTER_LFU_ODD_FIELD_READ_COMPLETE = 0,
	DISK_READ_THREAD_CENTER_LFU_EVEN_FIELD_READ_COMPLETE = 1,
	DISK_READ_THREAD_NEIGHBOR_LFU_READ_COMPLETE = 2,
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