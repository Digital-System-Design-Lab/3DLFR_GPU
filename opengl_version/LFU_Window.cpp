#include "LFU_Window.h"

LFU_Window::LFU_Window(LF_Config* config, const int& posX, const int& posY, DISK_READ_THREAD_STATE* disk_read_thread_state, bool use_window)
{
	_config = config;
	this->interlaced_LF_size = _config->LF_size / 2;
	this->LF_prefix = _config->path_LightField;
	curLFUID = getLFUID(posX, posY);
	this->state_disk_read_thread = disk_read_thread_state;
	
	printf("Allocating pinned memory");
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 4; j++) {
			printf(".");
			m_pinnedLFU[i][j] = alloc_uint8(interlaced_LF_size, "pinned");
		}
	}
	printf(" Complete\n");

	for (int i = 0; i < 12; i++) {
		if (i == 4 || i == 7) {
			m_row[i].odd_field = alloc_uint8(interlaced_LF_size, "pageable");
			m_row[i].even_field = nullptr;
			m_row[i].type = ROW;

			m_col[i].odd_field = alloc_uint8(interlaced_LF_size, "pageable");
			m_col[i].even_field = nullptr;
			m_col[i].type = COL;

			m_row[i].progress = LF_READ_PROGRESS_NOT_PREPARED;
			m_col[i].progress = LF_READ_PROGRESS_NOT_PREPARED;
		}
		else {
			m_row[i].odd_field = nullptr;
			m_row[i].even_field = nullptr;
			m_row[i].type = ROW;

			m_col[i].odd_field = nullptr;
			m_col[i].even_field = nullptr;
			m_col[i].type = COL;

			m_row[i].progress = LF_READ_PROGRESS_NOT_PREPARED;
			m_col[i].progress = LF_READ_PROGRESS_NOT_PREPARED;
		}
	}

	m_LFU[8].LF[FRONT] = &m_row[4]; // center
	m_LFU[8].LF[RIGHT] = &m_col[7];
	m_LFU[8].LF[BACK] = &m_row[7];
	m_LFU[8].LF[LEFT] = &m_col[4];

	m_LFU[8].LF[FRONT]->even_field = m_pinnedLFU[EVEN][FRONT];
	m_LFU[8].LF[RIGHT]->even_field = m_pinnedLFU[EVEN][RIGHT];
	m_LFU[8].LF[BACK]->even_field = m_pinnedLFU[EVEN][BACK];
	m_LFU[8].LF[LEFT]->even_field = m_pinnedLFU[EVEN][LEFT];

	printf("Reading Odd Field LFs");

	printf(".");

	int LF_num_each_dir[4];
	find_LF_number_BMW(LF_num_each_dir[0], LF_num_each_dir[1], LF_num_each_dir[2], LF_num_each_dir[3], curLFUID);

	for (int dir = 0; dir < 4; dir++) {
		m_LFU[8].id = curLFUID;
		if (m_LFU[8].LF[dir]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED) {
			read_uint8(m_LFU[8].LF[dir]->odd_field, BMW_LF[curLFUID][dir], ODD, posX, posY);
			m_LFU[8].LF[dir]->LF_number = LF_num_each_dir[dir];
			m_LFU[8].LF[dir]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
		}
	}
	m_center = &m_LFU[8];

	printf("Reading Even Field LFs");
	for (int dir = 0; dir < 4; dir++) {
		memcpy(m_pinnedLFU[ODD][dir], m_center->LF[dir]->odd_field, interlaced_LF_size);
		read_uint8(m_center->LF[dir]->even_field, BMW_LF[m_center->id][dir], EVEN, posX, posY);
		m_center->LF[dir]->progress = LF_READ_PROGRESS_EVEN_FIELD_PREPARED;
		printf(".");
	}
	pinned_memory_status = PINNED_LFU_EVEN_AVAILABLE;

	printf("Complete\n");
}

LFU_Window::LFU_Window(LF_Config* config, const int& posX, const int& posY, DISK_READ_THREAD_STATE* disk_read_thread_state)
{
	_config = config;
	this->interlaced_LF_size = _config->LF_size / 2;
	this->LF_prefix = _config->path_LightField;
	curLFUID = getLFUID(posX, posY);
	this->state_disk_read_thread = disk_read_thread_state;

	int LFUIDs[9];
	LFUIDs[N] = curLFUID + 1;
	LFUIDs[NE] = curLFUID + 56 + 1;
	LFUIDs[E] = curLFUID + 56;
	LFUIDs[SE] = curLFUID + 56 - 1;
	LFUIDs[S] = curLFUID - 1;
	LFUIDs[SW] = curLFUID - 56 - 1;
	LFUIDs[W] = curLFUID - 56;
	LFUIDs[NW] = curLFUID - 56 + 1;
	LFUIDs[8] = curLFUID;
	construct_window(interlaced_LF_size);

	printf("Reading Odd Field LFs");
	for (int i = 0; i < 9; i++)
	{
		printf(".");

		int LF_num_each_dir[4];
		find_LF_number_BMW(LF_num_each_dir[0], LF_num_each_dir[1], LF_num_each_dir[2], LF_num_each_dir[3], LFUIDs[i]);

		for (int dir = 0; dir < 4; dir++) {
			m_LFU[i].id = LFUIDs[i];
			if (m_LFU[i].LF[dir]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED) {
				read_uint8(m_LFU[i].LF[dir]->odd_field, BMW_LF[LFUIDs[i]][dir], ODD, posX, posY);
				m_LFU[i].LF[dir]->LF_number = LF_num_each_dir[dir];
				m_LFU[i].LF[dir]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
			}
		}
	}
	m_center = &m_LFU[8];

	printf("Reading Even Field LFs");
	for (int dir = 0; dir < 4; dir++) {
		memcpy(m_pinnedLFU[ODD][dir], m_center->LF[dir]->odd_field, interlaced_LF_size);
		read_uint8(m_center->LF[dir]->even_field, BMW_LF[m_center->id][dir], EVEN, posX, posY);
		m_center->LF[dir]->progress = LF_READ_PROGRESS_EVEN_FIELD_PREPARED;
		printf(".");
	}
	pinned_memory_status = PINNED_LFU_EVEN_AVAILABLE;

	printf("Complete\n");

	m_center->nbr[N] = &m_LFU[0];
	m_center->nbr[N]->nbr[S] = m_center;
	m_center->nbr[NE] = &m_LFU[1];
	m_center->nbr[NE]->nbr[SW] = m_center;
	m_center->nbr[E] = &m_LFU[2];
	m_center->nbr[E]->nbr[W] = m_center;
	m_center->nbr[SE] = &m_LFU[3];
	m_center->nbr[SE]->nbr[NW] = m_center;
	m_center->nbr[S] = &m_LFU[4];
	m_center->nbr[S]->nbr[N] = m_center;
	m_center->nbr[SW] = &m_LFU[5];
	m_center->nbr[SW]->nbr[NE] = m_center;
	m_center->nbr[W] = &m_LFU[6];
	m_center->nbr[W]->nbr[E] = m_center;
	m_center->nbr[NW] = &m_LFU[7];
	m_center->nbr[NW]->nbr[SE] = m_center;
}

LFU_Window::~LFU_Window()
{
	printf("Destruct LFU Window\n");
	for (int i = 0; i < 12; i++) {
		free_uint8(m_row[i].odd_field, "pageable");
		free_uint8(m_col[i].odd_field, "pageable");
	}

	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 4; j++) {
			free_uint8(m_pinnedLFU[i][j], "pinned");
		}
	}
}

void LFU_Window::construct_window(const size_t& light_field_size)
{
	printf("Allocating pinned memory");
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 4; j++) {
			printf(".");
			m_pinnedLFU[i][j] = alloc_uint8(light_field_size, "pinned");
		}
	}
	printf(" Complete\n");

	for (int i = 0; i < 12; i++) {
		m_row[i].odd_field = alloc_uint8(light_field_size, "pageable");
		m_row[i].even_field = nullptr;
		m_row[i].type = ROW;

		m_col[i].odd_field = alloc_uint8(light_field_size, "pageable");
		m_col[i].even_field = nullptr;
		m_col[i].type = COL;
			
		m_row[i].progress = LF_READ_PROGRESS_NOT_PREPARED;
		m_col[i].progress = LF_READ_PROGRESS_NOT_PREPARED;
	}

	m_LFU[0].LF[FRONT] = &m_row[1];
	m_LFU[0].LF[RIGHT] = &m_col[6];
	m_LFU[0].LF[BACK] = &m_row[4];
	m_LFU[0].LF[LEFT] = &m_col[3];

	m_LFU[1].LF[FRONT] = &m_row[2];
	m_LFU[1].LF[RIGHT] = &m_col[9];
	m_LFU[1].LF[BACK] = &m_row[5];
	m_LFU[1].LF[LEFT] = &m_col[6];

	m_LFU[2].LF[FRONT] = &m_row[5];
	m_LFU[2].LF[RIGHT] = &m_col[10];
	m_LFU[2].LF[BACK] = &m_row[8];
	m_LFU[2].LF[LEFT] = &m_col[7];

	m_LFU[3].LF[FRONT] = &m_row[8];
	m_LFU[3].LF[RIGHT] = &m_col[11];
	m_LFU[3].LF[BACK] = &m_row[11];
	m_LFU[3].LF[LEFT] = &m_col[8];

	m_LFU[4].LF[FRONT] = &m_row[7];
	m_LFU[4].LF[RIGHT] = &m_col[8];
	m_LFU[4].LF[BACK] = &m_row[10];
	m_LFU[4].LF[LEFT] = &m_col[5];

	m_LFU[5].LF[FRONT] = &m_row[6];
	m_LFU[5].LF[RIGHT] = &m_col[5];
	m_LFU[5].LF[BACK] = &m_row[9];
	m_LFU[5].LF[LEFT] = &m_col[2];

	m_LFU[6].LF[FRONT] = &m_row[3];
	m_LFU[6].LF[RIGHT] = &m_col[4];
	m_LFU[6].LF[BACK] = &m_row[6];
	m_LFU[6].LF[LEFT] = &m_col[1];

	m_LFU[7].LF[FRONT] = &m_row[0];
	m_LFU[7].LF[RIGHT] = &m_col[3];
	m_LFU[7].LF[BACK] = &m_row[3];
	m_LFU[7].LF[LEFT] = &m_col[0];

	m_LFU[8].LF[FRONT] = &m_row[4]; // center
	m_LFU[8].LF[RIGHT] = &m_col[7];
	m_LFU[8].LF[BACK] = &m_row[7];
	m_LFU[8].LF[LEFT] = &m_col[4];

	m_LFU[8].LF[FRONT]->even_field = m_pinnedLFU[EVEN][FRONT];
	m_LFU[8].LF[RIGHT]->even_field = m_pinnedLFU[EVEN][RIGHT];
	m_LFU[8].LF[BACK]->even_field = m_pinnedLFU[EVEN][BACK];
	m_LFU[8].LF[LEFT]->even_field = m_pinnedLFU[EVEN][LEFT];
}

int LFU_Window::update_window(const int& curPosX, const int& curPosY, const size_t& light_field_size, const MAIN_THREAD_STATE& main_thread_state)
{
	curLFUID = getLFUID(curPosX, curPosY);

	if (curLFUID != this->m_center->id || *state_disk_read_thread != DISK_READ_THREAD_NEIGHBOR_LFU_READ_COMPLETE) {
		switch (curLFUID - this->m_center->id) {
		case 57: {
			printf("[LFU_Window] Window Sliding - Upper Right\n");
		}break;
		case 55: {
			printf("[LFU_Window] Window Sliding - Lower Right\n");
		}break;
		case -57: {
			printf("[LFU_Window] Window Sliding - Lower Left\n");
		}break;
		case -55: {
			printf("[LFU_Window] Window Sliding - Upper Left\n");
		}break;

		case 56: {
			printf("[LFU_Window] Window Sliding - Right\n");
			*state_disk_read_thread = DISK_READ_THREAD_CENTER_LFU_READING;
			pinned_memory_status = PINNED_LFU_NOT_AVAILABLE;
			LFU* prevCenter = m_center;

			for (int i = 0; i < 4; i++) {
				m_center->nbr[NW]->LF[i]->progress = LF_READ_PROGRESS_NOT_PREPARED;
				m_center->nbr[W]->LF[i]->progress = LF_READ_PROGRESS_NOT_PREPARED;
				m_center->nbr[SW]->LF[i]->progress = LF_READ_PROGRESS_NOT_PREPARED;
			}

			m_center->nbr[E]->nbr[N] = m_center->nbr[NE];
			m_center->nbr[E]->nbr[NE] = m_center->nbr[NW];
			m_center->nbr[E]->nbr[E] = m_center->nbr[W];
			m_center->nbr[E]->nbr[SE] = m_center->nbr[SW];
			m_center->nbr[E]->nbr[S] = m_center->nbr[SE];
			m_center->nbr[E]->nbr[SW] = m_center->nbr[S];
			m_center->nbr[E]->nbr[W] = m_center;
			m_center->nbr[E]->nbr[NW] = m_center->nbr[N];
			m_center = m_center->nbr[E];

			m_center->nbr[NE]->LF[LEFT] = m_center->nbr[N]->LF[RIGHT]; // LF 재사용
			m_center->nbr[E]->LF[LEFT] = m_center->LF[RIGHT];
			m_center->nbr[SE]->LF[LEFT] = m_center->nbr[S]->LF[RIGHT];

			m_center->nbr[NE]->id = m_center->nbr[N]->id + 56; // id 업데이트
			m_center->nbr[E]->id = m_center->id + 56;
			m_center->nbr[SE]->id = m_center->nbr[S]->id + 56;

			int f, r, b, l;
			find_LF_number_BMW(f, r, b, l, m_center->nbr[NE]->id); // LF number 업데이트
			m_center->nbr[NE]->LF[FRONT]->LF_number = f;
			m_center->nbr[NE]->LF[RIGHT]->LF_number = r;
			m_center->nbr[NE]->LF[BACK]->LF_number = b;
			m_center->nbr[NE]->LF[LEFT]->LF_number = l;

			find_LF_number_BMW(f, r, b, l, m_center->nbr[E]->id);
			m_center->nbr[E]->LF[FRONT]->LF_number = f;
			m_center->nbr[E]->LF[RIGHT]->LF_number = r;
			m_center->nbr[E]->LF[BACK]->LF_number = b;
			m_center->nbr[E]->LF[LEFT]->LF_number = l;

			find_LF_number_BMW(f, r, b, l, m_center->nbr[SE]->id);
			m_center->nbr[SE]->LF[FRONT]->LF_number = f;
			m_center->nbr[SE]->LF[RIGHT]->LF_number = r;
			m_center->nbr[SE]->LF[BACK]->LF_number = b;
			m_center->nbr[SE]->LF[LEFT]->LF_number = l;

			// prev center's pinned even memory address goes to null while new center gets it
			uint8_t* even_fields[4];
			for (int dir = 0; dir < 4; dir++) {
				even_fields[dir] = prevCenter->LF[dir]->even_field;
				prevCenter->LF[dir]->even_field = nullptr;
			}
			for (int dir = 0; dir < 4; dir++) {
				m_center->LF[dir]->even_field = even_fields[dir];
				if (m_center->LF[dir]->progress == LF_READ_PROGRESS_EVEN_FIELD_PREPARED)
					m_center->LF[dir]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
			}
		} break;
		case -56: {
			// LF[LEFT] - nbr[NW], nbr[W], nbr[SW] should be replaced
			printf("[LFU_Window] Window Sliding - Left\n");
			*state_disk_read_thread = DISK_READ_THREAD_CENTER_LFU_READING;
			pinned_memory_status = PINNED_LFU_NOT_AVAILABLE;
			LFU* prevCenter = m_center;

			for (int i = 0; i < 4; i++) {
				m_center->nbr[NE]->LF[i]->progress = LF_READ_PROGRESS_NOT_PREPARED;
				m_center->nbr[E]->LF[i]->progress = LF_READ_PROGRESS_NOT_PREPARED;
				m_center->nbr[SE]->LF[i]->progress = LF_READ_PROGRESS_NOT_PREPARED;
			}

			m_center->nbr[W]->nbr[N] = m_center->nbr[NW]; // 새로운 LFU 관계 정의 (LFU Window Sliding)
			m_center->nbr[W]->nbr[NE] = m_center->nbr[N];
			m_center->nbr[W]->nbr[E] = m_center;
			m_center->nbr[W]->nbr[SE] = m_center->nbr[S];
			m_center->nbr[W]->nbr[S] = m_center->nbr[SW];
			m_center->nbr[W]->nbr[SW] = m_center->nbr[SE];
			m_center->nbr[W]->nbr[W] = m_center->nbr[E];
			m_center->nbr[W]->nbr[NW] = m_center->nbr[NE];
			m_center = m_center->nbr[W];

			m_center->nbr[NW]->LF[RIGHT] = m_center->nbr[N]->LF[LEFT]; // LF 재사용
			m_center->nbr[W]->LF[RIGHT] = m_center->LF[LEFT];
			m_center->nbr[SW]->LF[RIGHT] = m_center->nbr[S]->LF[LEFT];

			m_center->nbr[NW]->id = m_center->nbr[N]->id - 56;
			m_center->nbr[W]->id = m_center->id - 56;
			m_center->nbr[SW]->id = m_center->nbr[S]->id - 56; // id 업데이트

			int f, r, b, l;
			find_LF_number_BMW(f, r, b, l, m_center->nbr[SW]->id); // LF number 업데이트
			m_center->nbr[SW]->LF[FRONT]->LF_number = f;
			m_center->nbr[SW]->LF[RIGHT]->LF_number = r;
			m_center->nbr[SW]->LF[BACK]->LF_number = b;
			m_center->nbr[SW]->LF[LEFT]->LF_number = l;

			find_LF_number_BMW(f, r, b, l, m_center->nbr[W]->id);
			m_center->nbr[W]->LF[FRONT]->LF_number = f;
			m_center->nbr[W]->LF[RIGHT]->LF_number = r;
			m_center->nbr[W]->LF[BACK]->LF_number = b;
			m_center->nbr[W]->LF[LEFT]->LF_number = l;

			find_LF_number_BMW(f, r, b, l, m_center->nbr[NW]->id);
			m_center->nbr[NW]->LF[FRONT]->LF_number = f;
			m_center->nbr[NW]->LF[RIGHT]->LF_number = r;
			m_center->nbr[NW]->LF[BACK]->LF_number = b;
			m_center->nbr[NW]->LF[LEFT]->LF_number = l;
			
			// prev center's pinned even memory address goes to null while new center gets it
			uint8_t* even_fields[4];
			for (int dir = 0; dir < 4; dir++) {
				even_fields[dir] = prevCenter->LF[dir]->even_field;
				prevCenter->LF[dir]->even_field = nullptr;
			}
			for (int dir = 0; dir < 4; dir++) {
				m_center->LF[dir]->even_field = even_fields[dir];
				if (m_center->LF[dir]->progress == LF_READ_PROGRESS_EVEN_FIELD_PREPARED)
					m_center->LF[dir]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
			}
		} break;
		case 1: {
			printf("[LFU_Window] Window Sliding - Up\n");
			*state_disk_read_thread = DISK_READ_THREAD_CENTER_LFU_READING;
			pinned_memory_status = PINNED_LFU_NOT_AVAILABLE;
			LFU* prevCenter = m_center;

			for (int i = 0; i < 4; i++) {
				m_center->nbr[SW]->LF[i]->progress = LF_READ_PROGRESS_NOT_PREPARED;
				m_center->nbr[S]->LF[i]->progress = LF_READ_PROGRESS_NOT_PREPARED;
				m_center->nbr[SE]->LF[i]->progress = LF_READ_PROGRESS_NOT_PREPARED;
			}

			m_center->nbr[N]->nbr[N] = m_center->nbr[S];
			m_center->nbr[N]->nbr[NE] = m_center->nbr[SE];
			m_center->nbr[N]->nbr[E] = m_center->nbr[NE];
			m_center->nbr[N]->nbr[SE] = m_center->nbr[E];
			m_center->nbr[N]->nbr[S] = m_center;
			m_center->nbr[N]->nbr[SW] = m_center->nbr[W];
			m_center->nbr[N]->nbr[W] = m_center->nbr[NW];
			m_center->nbr[N]->nbr[NW] = m_center->nbr[SW];
			m_center = m_center->nbr[N];

			m_center->nbr[NE]->LF[BACK] = m_center->nbr[E]->LF[FRONT]; // LF 재사용
			m_center->nbr[N]->LF[BACK] = m_center->LF[FRONT];
			m_center->nbr[NW]->LF[BACK] = m_center->nbr[W]->LF[FRONT];

			m_center->nbr[NE]->id = m_center->nbr[E]->id + 1;
			m_center->nbr[N]->id = m_center->id + 1;
			m_center->nbr[NW]->id = m_center->nbr[W]->id + 1;

			int f, r, b, l;
			find_LF_number_BMW(f, r, b, l, m_center->nbr[N]->id); // LF number 업데이트
			m_center->nbr[N]->LF[FRONT]->LF_number = f;
			m_center->nbr[N]->LF[RIGHT]->LF_number = r;
			m_center->nbr[N]->LF[BACK]->LF_number = b;
			m_center->nbr[N]->LF[LEFT]->LF_number = l;

			find_LF_number_BMW(f, r, b, l, m_center->nbr[NE]->id);
			m_center->nbr[NE]->LF[FRONT]->LF_number = f;
			m_center->nbr[NE]->LF[RIGHT]->LF_number = r;
			m_center->nbr[NE]->LF[BACK]->LF_number = b;
			m_center->nbr[NE]->LF[LEFT]->LF_number = l;

			find_LF_number_BMW(f, r, b, l, m_center->nbr[NW]->id);
			m_center->nbr[NW]->LF[FRONT]->LF_number = f;
			m_center->nbr[NW]->LF[RIGHT]->LF_number = r;
			m_center->nbr[NW]->LF[BACK]->LF_number = b;
			m_center->nbr[NW]->LF[LEFT]->LF_number = l;

			// prev center's pinned even memory address goes to null while new center gets it
			uint8_t* even_fields[4];
			for (int dir = 0; dir < 4; dir++) {
				even_fields[dir] = prevCenter->LF[dir]->even_field;
				prevCenter->LF[dir]->even_field = nullptr;
			}
			for (int dir = 0; dir < 4; dir++) {
				m_center->LF[dir]->even_field = even_fields[dir];
				if (m_center->LF[dir]->progress == LF_READ_PROGRESS_EVEN_FIELD_PREPARED)
					m_center->LF[dir]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
			}
		} break;
		case -1: {
			printf("[LFU_Window] Window Sliding - Down\n");
			*state_disk_read_thread = DISK_READ_THREAD_CENTER_LFU_READING;
			pinned_memory_status = PINNED_LFU_NOT_AVAILABLE;
			LFU* prevCenter = m_center;

			for (int i = 0; i < 4; i++) {
				m_center->nbr[NW]->LF[i]->progress = LF_READ_PROGRESS_NOT_PREPARED;
				m_center->nbr[N]->LF[i]->progress = LF_READ_PROGRESS_NOT_PREPARED;
				m_center->nbr[NE]->LF[i]->progress = LF_READ_PROGRESS_NOT_PREPARED;
			}

			m_center->nbr[S]->nbr[N] = m_center;
			m_center->nbr[S]->nbr[NE] = m_center->nbr[E];
			m_center->nbr[S]->nbr[E] = m_center->nbr[SE];
			m_center->nbr[S]->nbr[SE] = m_center->nbr[NE];
			m_center->nbr[S]->nbr[S] = m_center->nbr[N];
			m_center->nbr[S]->nbr[SW] = m_center->nbr[NW];
			m_center->nbr[S]->nbr[W] = m_center->nbr[SW];
			m_center->nbr[S]->nbr[NW] = m_center->nbr[W];
			m_center = m_center->nbr[S];

			m_center->nbr[SE]->LF[FRONT] = m_center->nbr[E]->LF[BACK]; // LF 재사용
			m_center->nbr[S]->LF[FRONT] = m_center->LF[BACK];
			m_center->nbr[SW]->LF[FRONT] = m_center->nbr[W]->LF[BACK];

			m_center->nbr[NE]->id = m_center->nbr[E]->id + 1;
			m_center->nbr[N]->id = m_center->id + 1;
			m_center->nbr[NW]->id = m_center->nbr[W]->id + 1;

			m_center->nbr[SW]->id = m_center->nbr[W]->id - 1;
			m_center->nbr[S]->id = m_center->id - 1;
			m_center->nbr[SE]->id = m_center->nbr[E]->id - 1;

			int f, r, b, l;
			find_LF_number_BMW(f, r, b, l, m_center->nbr[SW]->id); // LF number 업데이트
			m_center->nbr[SW]->LF[FRONT]->LF_number = f;
			m_center->nbr[SW]->LF[RIGHT]->LF_number = r;
			m_center->nbr[SW]->LF[BACK]->LF_number = b;
			m_center->nbr[SW]->LF[LEFT]->LF_number = l;

			find_LF_number_BMW(f, r, b, l, m_center->nbr[S]->id);
			m_center->nbr[S]->LF[FRONT]->LF_number = f;
			m_center->nbr[S]->LF[RIGHT]->LF_number = r;
			m_center->nbr[S]->LF[BACK]->LF_number = b;
			m_center->nbr[S]->LF[LEFT]->LF_number = l;

			find_LF_number_BMW(f, r, b, l, m_center->nbr[SE]->id);
			m_center->nbr[SE]->LF[FRONT]->LF_number = f;
			m_center->nbr[SE]->LF[RIGHT]->LF_number = r;
			m_center->nbr[SE]->LF[BACK]->LF_number = b;
			m_center->nbr[SE]->LF[LEFT]->LF_number = l;

			// prev center's pinned even memory address goes to null while new center gets it
			uint8_t* even_fields[4];
			for (int dir = 0; dir < 4; dir++) {
				even_fields[dir] = prevCenter->LF[dir]->even_field;
				prevCenter->LF[dir]->even_field = nullptr;
			}
			for (int dir = 0; dir < 4; dir++) {
				m_center->LF[dir]->even_field = even_fields[dir];
				if (m_center->LF[dir]->progress == LF_READ_PROGRESS_EVEN_FIELD_PREPARED)
					m_center->LF[dir]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
			}
		} break;
		}

		// black lines (center)
		if (m_center->LF[FRONT]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
			if(read_uint8(m_center->LF[FRONT]->odd_field, BMW_LF[m_center->id][FRONT], ODD, curPosX, curPosY) < 0) return -1;
			m_center->LF[FRONT]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
		}
		if (m_center->LF[RIGHT]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_uint8(m_center->LF[RIGHT]->odd_field, BMW_LF[m_center->id][RIGHT], ODD, curPosX, curPosY) < 0) return -1;
			m_center->LF[RIGHT]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
		}
		if (m_center->LF[BACK]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_uint8(m_center->LF[BACK]->odd_field, BMW_LF[m_center->id][BACK], ODD, curPosX, curPosY) < 0) return -1;
			m_center->LF[BACK]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
		}
		if (m_center->LF[LEFT]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_uint8(m_center->LF[LEFT]->odd_field, BMW_LF[m_center->id][LEFT], ODD, curPosX, curPosY) < 0) return -1;
			m_center->LF[LEFT]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
		}
		*state_disk_read_thread = DISK_READ_THREAD_CENTER_LFU_READ_COMPLETE;
		if (pinned_memory_status == PINNED_LFU_NOT_AVAILABLE) {
			memcpy(m_pinnedLFU[ODD][FRONT], m_center->LF[FRONT]->odd_field, light_field_size);
			memcpy(m_pinnedLFU[ODD][RIGHT], m_center->LF[RIGHT]->odd_field, light_field_size);
			memcpy(m_pinnedLFU[ODD][BACK], m_center->LF[BACK]->odd_field, light_field_size);
			memcpy(m_pinnedLFU[ODD][LEFT], m_center->LF[LEFT]->odd_field, light_field_size);
		}
		pinned_memory_status = PINNED_LFU_ODD_AVAILABLE;

		for (int dir = 0; dir < 4; dir++) {
			if (m_center->LF[dir]->progress < LF_READ_PROGRESS_EVEN_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
				if (read_uint8(m_center->LF[dir]->even_field, BMW_LF[m_center->id][dir], EVEN, curPosX, curPosY) < 0) return -1;
				m_center->LF[dir]->progress = LF_READ_PROGRESS_EVEN_FIELD_PREPARED;
			}
			if (dir == 3) {
				pinned_memory_status = PINNED_LFU_EVEN_AVAILABLE;
			}
		}
		*state_disk_read_thread = DISK_READ_THREAD_NEIGHBOR_LFU_READING;
		// green lines
		if (m_center->nbr[N]->LF[LEFT]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_uint8(m_center->nbr[N]->LF[LEFT]->odd_field, BMW_LF[m_center->nbr[N]->id][LEFT], ODD, curPosX, curPosY) < 0) return -1;
			m_center->nbr[N]->LF[LEFT]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
		}
		if (m_center->nbr[N]->LF[RIGHT]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_uint8(m_center->nbr[N]->LF[RIGHT]->odd_field, BMW_LF[m_center->nbr[N]->id][RIGHT], ODD, curPosX, curPosY) < 0) return -1;
			m_center->nbr[N]->LF[RIGHT]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
		}
		if (m_center->nbr[S]->LF[LEFT]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_uint8(m_center->nbr[S]->LF[LEFT]->odd_field, BMW_LF[m_center->nbr[S]->id][LEFT], ODD, curPosX, curPosY) < 0) return -1;
			m_center->nbr[S]->LF[LEFT]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
		}
		if (m_center->nbr[S]->LF[RIGHT]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_uint8(m_center->nbr[S]->LF[RIGHT]->odd_field, BMW_LF[m_center->nbr[S]->id][RIGHT], ODD, curPosX, curPosY) < 0) return -1;
			m_center->nbr[S]->LF[RIGHT]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
		}
		if (m_center->nbr[E]->LF[FRONT]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_uint8(m_center->nbr[E]->LF[FRONT]->odd_field, BMW_LF[m_center->nbr[E]->id][FRONT], ODD, curPosX, curPosY) < 0) return -1;
			m_center->nbr[E]->LF[FRONT]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
		}
		if (m_center->nbr[E]->LF[BACK]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_uint8(m_center->nbr[E]->LF[BACK]->odd_field, BMW_LF[m_center->nbr[E]->id][BACK], ODD, curPosX, curPosY) < 0) return -1;
			m_center->nbr[E]->LF[BACK]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
		}
		if (m_center->nbr[W]->LF[FRONT]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_uint8(m_center->nbr[W]->LF[FRONT]->odd_field, BMW_LF[m_center->nbr[W]->id][FRONT], ODD, curPosX, curPosY) < 0) return -1;
			m_center->nbr[W]->LF[FRONT]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
		}
		if (m_center->nbr[W]->LF[BACK]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_uint8(m_center->nbr[W]->LF[BACK]->odd_field, BMW_LF[m_center->nbr[W]->id][BACK], ODD, curPosX, curPosY) < 0) return -1;
			m_center->nbr[W]->LF[BACK]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
		}

		// red lines
		if (m_center->nbr[N]->LF[FRONT]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_uint8(m_center->nbr[N]->LF[FRONT]->odd_field, BMW_LF[m_center->nbr[N]->id][FRONT], ODD, curPosX, curPosY) < 0) return -1;
			m_center->nbr[N]->LF[FRONT]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
		}
		if (m_center->nbr[E]->LF[RIGHT]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_uint8(m_center->nbr[E]->LF[RIGHT]->odd_field, BMW_LF[m_center->nbr[E]->id][RIGHT], ODD, curPosX, curPosY) < 0) return -1;
			m_center->nbr[E]->LF[RIGHT]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
		}
		if (m_center->nbr[S]->LF[BACK]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_uint8(m_center->nbr[S]->LF[BACK]->odd_field, BMW_LF[m_center->nbr[S]->id][BACK], ODD, curPosX, curPosY) < 0) return -1;
			m_center->nbr[S]->LF[BACK]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
		}
		if (m_center->nbr[W]->LF[LEFT]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_uint8(m_center->nbr[W]->LF[LEFT]->odd_field, BMW_LF[m_center->nbr[W]->id][LEFT], ODD, curPosX, curPosY) < 0) return -1;
			m_center->nbr[W]->LF[LEFT]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
		}

		// blue lines
		if (m_center->nbr[NE]->LF[FRONT]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_uint8(m_center->nbr[NE]->LF[FRONT]->odd_field, BMW_LF[m_center->nbr[NE]->id][FRONT], ODD, curPosX, curPosY) < 0) return -1;
			m_center->nbr[NE]->LF[FRONT]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
		}
		if (m_center->nbr[SE]->LF[BACK]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_uint8(m_center->nbr[SE]->LF[BACK]->odd_field, BMW_LF[m_center->nbr[SE]->id][BACK], ODD, curPosX, curPosY) < 0) return -1;
			m_center->nbr[SE]->LF[BACK]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
		}
		if (m_center->nbr[SW]->LF[BACK]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_uint8(m_center->nbr[SW]->LF[BACK]->odd_field, BMW_LF[m_center->nbr[SW]->id][BACK], ODD, curPosX, curPosY) < 0) return -1;
			m_center->nbr[SW]->LF[BACK]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
		}
		if (m_center->nbr[NW]->LF[FRONT]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_uint8(m_center->nbr[NW]->LF[FRONT]->odd_field, BMW_LF[m_center->nbr[NW]->id][FRONT], ODD, curPosX, curPosY) < 0) return -1;
			m_center->nbr[NW]->LF[FRONT]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
		}
		if (m_center->nbr[NE]->LF[RIGHT]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_uint8(m_center->nbr[NE]->LF[RIGHT]->odd_field, BMW_LF[m_center->nbr[NE]->id][RIGHT], ODD, curPosX, curPosY) < 0) return -1;
			m_center->nbr[NE]->LF[RIGHT]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
		}
		if (m_center->nbr[SE]->LF[RIGHT]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_uint8(m_center->nbr[SE]->LF[RIGHT]->odd_field, BMW_LF[m_center->nbr[SE]->id][RIGHT], ODD, curPosX, curPosY) < 0) return -1;
			m_center->nbr[SE]->LF[RIGHT]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
		}
		if (m_center->nbr[SW]->LF[LEFT]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_uint8(m_center->nbr[SW]->LF[LEFT]->odd_field, BMW_LF[m_center->nbr[SW]->id][LEFT], ODD, curPosX, curPosY) < 0) return -1;
			m_center->nbr[SW]->LF[LEFT]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
		}
		if (m_center->nbr[NW]->LF[LEFT]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_uint8(m_center->nbr[NW]->LF[LEFT]->odd_field, BMW_LF[m_center->nbr[NW]->id][LEFT], ODD, curPosX, curPosY) < 0) return -1;
			m_center->nbr[NW]->LF[LEFT]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
		}

		*state_disk_read_thread = DISK_READ_THREAD_NEIGHBOR_LFU_READ_COMPLETE;
		printf("window update done\n");
	}

	return 0;
}

size_t LFU_Window::read_uint8(uint8_t* buf, std::string filename, const INTERLACE_FIELD& field, const int& curPosX, const int& curPosY, size_t size) {
	size_t ret;
	filename = LF_prefix + filename;
	if (field == ODD) filename += "_odd.bgr";
	else filename += "_even.bgr";

	FILE* fp = fopen(filename.c_str(), "rb");
	ret = fp == nullptr ? -1 : 0;
	if (ret < 0) {
		printf("open failed, %s\n", filename.c_str());
		assert(ret == 0);
		exit(1);
	}
	if (size == 0) {
		size = this->interlaced_LF_size;
	}
	const size_t num_chunk = 50;
	const size_t chunk_size = size / num_chunk;
	size_t next_chunk_begin = 0;
	for (int i = 0; i < num_chunk; i++) {
		if (curLFUID != getLFUID(curPosX, curPosY)) {
			*state_disk_read_thread = DISK_READ_THREAD_CENTER_LFU_READING;
			printf("LF read break\n");
			return -1; // Interrupt
		}
		ret += fread(buf + next_chunk_begin, 1, sizeof(uint8_t) * chunk_size, fp); // x64
		next_chunk_begin += chunk_size;
	}
	fclose(fp);

	if (ret != size) {
		printf("read failed, %s\n", filename.c_str());
		assert(ret == size);
		exit(1);
	}

	return ret;
}