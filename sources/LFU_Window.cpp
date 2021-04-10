#include "LFU_Window.h"
#include <algorithm>

LFU_Window::LFU_Window(LF_Config* config, const int& posX, const int& posY, DISK_READ_THREAD_STATE* disk_read_thread_state, bool use_window)
{
	this->use_window = use_window;
	_config = config;
	this->interlaced_LF_size = _config->LF_size / 2;
	this->LF_prefix = _config->path_LightField;
	curLFUID = getLFUID(posX, posY);
	this->state_disk_read_thread = disk_read_thread_state;
	
	if (use_window)
	{
		priority_LFU_read.resize(8);
		set_LFU_read_priority(posX, posY);

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
				
				if (!isLFReadCompleted(m_LFU[i].LF[dir], ODD)) {
					read_LF(m_LFU[i].LF[dir], BMW_LF[LFUIDs[i]][dir], ODD, posX, posY);
					m_LFU[i].LF[dir]->LF_number = LF_num_each_dir[dir];
				}
			}
		}
		m_center = &m_LFU[8];

		printf("Reading Even Field LFs");
		for (int dir = 0; dir < 4; dir++) {
			read_LF(m_center->LF[dir], BMW_LF[m_center->id][dir], EVEN, posX, posY);
			printf(".");
		}
		printf("Complete\n");

		m_center->nbrLFU[N] = &m_LFU[0];
		m_center->nbrLFU[N]->nbrLFU[S] = m_center;
		m_center->nbrLFU[NE] = &m_LFU[1];
		m_center->nbrLFU[NE]->nbrLFU[SW] = m_center;
		m_center->nbrLFU[E] = &m_LFU[2];
		m_center->nbrLFU[E]->nbrLFU[W] = m_center;
		m_center->nbrLFU[SE] = &m_LFU[3];
		m_center->nbrLFU[SE]->nbrLFU[NW] = m_center;
		m_center->nbrLFU[S] = &m_LFU[4];
		m_center->nbrLFU[S]->nbrLFU[N] = m_center;
		m_center->nbrLFU[SW] = &m_LFU[5];
		m_center->nbrLFU[SW]->nbrLFU[NE] = m_center;
		m_center->nbrLFU[W] = &m_LFU[6];
		m_center->nbrLFU[W]->nbrLFU[E] = m_center;
		m_center->nbrLFU[NW] = &m_LFU[7];
		m_center->nbrLFU[NW]->nbrLFU[SE] = m_center;

		*state_disk_read_thread = DISK_READ_THREAD_NEIGHBOR_LFU_READ_COMPLETE;
	}
	else {
		printf("Allocating pinned memory...");
		m_row[4].odd_field = alloc_uint8(interlaced_LF_size, "pinned");
		m_row[4].even_field = alloc_uint8(interlaced_LF_size, "pinned");
		m_row[4].type = ROW;

		m_row[7].odd_field = alloc_uint8(interlaced_LF_size, "pinned");
		m_row[7].even_field = alloc_uint8(interlaced_LF_size, "pinned");
		m_row[7].type = ROW;

		m_col[4].odd_field = alloc_uint8(interlaced_LF_size, "pinned");
		m_col[4].even_field = alloc_uint8(interlaced_LF_size, "pinned");
		m_col[4].type = COL;

		m_col[7].odd_field = alloc_uint8(interlaced_LF_size, "pinned");
		m_col[7].even_field = alloc_uint8(interlaced_LF_size, "pinned");
		m_col[7].type = COL;
		printf(" Complete\n");

		m_LFU[8].LF[FRONT] = &m_row[4]; // center
		m_LFU[8].LF[RIGHT] = &m_col[7];
		m_LFU[8].LF[BACK] = &m_row[7];
		m_LFU[8].LF[LEFT] = &m_col[4];
		m_center = &m_LFU[8];

		printf("Reading Odd Field LFs");
		int LF_num_each_dir[4];
		find_LF_number_BMW(LF_num_each_dir[0], LF_num_each_dir[1], LF_num_each_dir[2], LF_num_each_dir[3], curLFUID);

		for (int dir = 0; dir < 4; dir++) 
		{
			m_center->id = curLFUID;
			if (!isLFReadCompleted(m_center->LF[dir], ODD))
			{
				read_LF(m_center->LF[dir], BMW_LF[curLFUID][dir], ODD, posX, posY);
				read_LF(m_center->LF[dir], BMW_LF[curLFUID][dir], EVEN, posX, posY);
				m_center->LF[dir]->LF_number = LF_num_each_dir[dir];
			}
		}
		*state_disk_read_thread = DISK_READ_THREAD_CENTER_LFU_EVEN_FIELD_READ_COMPLETE;
	}
	printf("Complete\n");
}

LFU_Window::~LFU_Window()
{
	printf("Destruct LFU Window\n");
	if (this->use_window) {
		for (int i = 0; i < 12; i++) {
			free_uint8(m_row[i].odd_field, "pinned");
			free_uint8(m_col[i].odd_field, "pinned");
		}
	}
	else
	{
		free_uint8(m_row[4].odd_field, "pinned");
		free_uint8(m_row[4].even_field, "pinned");
		free_uint8(m_row[7].odd_field, "pinned");
		free_uint8(m_row[7].even_field, "pinned");
		free_uint8(m_col[4].odd_field, "pinned");
		free_uint8(m_col[4].even_field, "pinned");
		free_uint8(m_col[7].odd_field, "pinned");
		free_uint8(m_col[7].even_field, "pinned");
	}
}

void LFU_Window::construct_window(const size_t& light_field_size)
{
	printf("Allocating pinned memory");
	for (int i = 0; i < 12; i++) {
		m_row[i].odd_field = alloc_uint8(light_field_size, "pinned");
		if(i == 4 || i == 7) m_row[i].even_field = alloc_uint8(light_field_size, "pinned");
		else m_row[i].even_field = nullptr;
		m_row[i].type = ROW;

		m_col[i].odd_field = alloc_uint8(light_field_size, "pinned");
		if (i == 4 || i == 7) m_col[i].even_field = alloc_uint8(light_field_size, "pinned");
		else m_col[i].even_field = nullptr;
		m_col[i].type = COL;
		printf(".");
	} printf(" completed\n");

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
}

void LFU_Window::update_LFU(const int& curPosX, const int& curPosY)
{
	curLFUID = getLFUID(curPosX, curPosY);

	if (curLFUID != this->m_center->id)
	{
		*state_disk_read_thread = DISK_READ_THREAD_CENTER_LFU_READING;
		int f, r, b, l;
		find_LF_number_BMW(f, r, b, l, curLFUID); // LF number 업데이트
		switch (curLFUID - this->m_center->id) 
		{
		case 56: // right
		{
			m_center->id = curLFUID;
			std::swap(m_center->LF[LEFT], m_center->LF[RIGHT]);
			m_center->LF[FRONT]->LF_number = f;
			m_center->LF[RIGHT]->LF_number =r;
			m_center->LF[BACK]->LF_number = b;
			read_LF(m_center->LF[FRONT], BMW_LF[m_center->id][FRONT], ODD, curPosX, curPosY);
			read_LF(m_center->LF[FRONT], BMW_LF[m_center->id][FRONT], EVEN, curPosX, curPosY);
			read_LF(m_center->LF[RIGHT], BMW_LF[m_center->id][RIGHT], ODD, curPosX, curPosY);
			read_LF(m_center->LF[RIGHT], BMW_LF[m_center->id][RIGHT], EVEN, curPosX, curPosY);
			read_LF(m_center->LF[BACK], BMW_LF[m_center->id][BACK], ODD, curPosX, curPosY);
			read_LF(m_center->LF[BACK], BMW_LF[m_center->id][BACK], EVEN, curPosX, curPosY);
		} break;
		case -56: // left
		{
			m_center->id = curLFUID;
			std::swap(m_center->LF[LEFT], m_center->LF[RIGHT]);
			m_center->LF[FRONT]->LF_number = f;
			m_center->LF[LEFT]->LF_number = l;
			m_center->LF[BACK]->LF_number = b;
			read_LF(m_center->LF[FRONT], BMW_LF[m_center->id][FRONT], ODD, curPosX, curPosY);
			read_LF(m_center->LF[FRONT], BMW_LF[m_center->id][FRONT], EVEN, curPosX, curPosY);
			read_LF(m_center->LF[LEFT], BMW_LF[m_center->id][LEFT], ODD, curPosX, curPosY);
			read_LF(m_center->LF[LEFT], BMW_LF[m_center->id][LEFT], EVEN, curPosX, curPosY);
			read_LF(m_center->LF[BACK], BMW_LF[m_center->id][BACK], ODD, curPosX, curPosY);
			read_LF(m_center->LF[BACK], BMW_LF[m_center->id][BACK], EVEN, curPosX, curPosY);
		} break;

		case 1: // up
		{
			m_center->id = curLFUID;
			std::swap(m_center->LF[FRONT], m_center->LF[BACK]);
			m_center->LF[FRONT]->LF_number = f;
			m_center->LF[RIGHT]->LF_number = r;
			m_center->LF[LEFT]->LF_number = l;
			read_LF(m_center->LF[FRONT], BMW_LF[m_center->id][FRONT], ODD, curPosX, curPosY);
			read_LF(m_center->LF[FRONT], BMW_LF[m_center->id][FRONT], EVEN, curPosX, curPosY);
			read_LF(m_center->LF[RIGHT], BMW_LF[m_center->id][RIGHT], ODD, curPosX, curPosY);
			read_LF(m_center->LF[RIGHT], BMW_LF[m_center->id][RIGHT], EVEN, curPosX, curPosY);
			read_LF(m_center->LF[LEFT], BMW_LF[m_center->id][LEFT], ODD, curPosX, curPosY);
			read_LF(m_center->LF[LEFT], BMW_LF[m_center->id][LEFT], EVEN, curPosX, curPosY);
		} break;
		case -1: // down
		{
			m_center->id = curLFUID;
			std::swap(m_center->LF[FRONT], m_center->LF[BACK]);
			m_center->LF[BACK]->LF_number = b;
			m_center->LF[RIGHT]->LF_number = r;
			m_center->LF[LEFT]->LF_number = l;
			read_LF(m_center->LF[BACK], BMW_LF[m_center->id][BACK], ODD, curPosX, curPosY);
			read_LF(m_center->LF[BACK], BMW_LF[m_center->id][BACK], EVEN, curPosX, curPosY);
			read_LF(m_center->LF[RIGHT], BMW_LF[m_center->id][RIGHT], ODD, curPosX, curPosY);
			read_LF(m_center->LF[RIGHT], BMW_LF[m_center->id][RIGHT], EVEN, curPosX, curPosY);
			read_LF(m_center->LF[LEFT], BMW_LF[m_center->id][LEFT], ODD, curPosX, curPosY);
			read_LF(m_center->LF[LEFT], BMW_LF[m_center->id][LEFT], EVEN, curPosX, curPosY);
		} break;
		}
	}

	*state_disk_read_thread = DISK_READ_THREAD_CENTER_LFU_EVEN_FIELD_READ_COMPLETE;
}

int LFU_Window::update_window(const int& curPosX, const int& curPosY, const size_t& light_field_size, const MAIN_THREAD_STATE& main_thread_state)
{
	curLFUID = getLFUID(curPosX, curPosY);
	
	if (curLFUID != this->m_center->id)
	{
		*state_disk_read_thread = DISK_READ_THREAD_CENTER_LFU_READING;
		LFU* prevCenter = m_center;

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
			m_center = prevCenter->nbrLFU[E];
			m_center->nbrLFU[N] = prevCenter->nbrLFU[NE];
			m_center->nbrLFU[NE] = prevCenter->nbrLFU[NW];
			m_center->nbrLFU[E] = prevCenter->nbrLFU[W];
			m_center->nbrLFU[SE] = prevCenter->nbrLFU[SW];
			m_center->nbrLFU[S] = prevCenter->nbrLFU[SE];
			m_center->nbrLFU[SW] = prevCenter->nbrLFU[S];
			m_center->nbrLFU[W] = prevCenter;
			m_center->nbrLFU[NW] = prevCenter->nbrLFU[N];

			std::swap(m_center->nbrLFU[NE]->LF[RIGHT], m_center->nbrLFU[NE]->LF[LEFT]);
			std::swap(m_center->nbrLFU[E]->LF[RIGHT], m_center->nbrLFU[E]->LF[LEFT]);
			std::swap(m_center->nbrLFU[SE]->LF[RIGHT], m_center->nbrLFU[SE]->LF[LEFT]);
			for (int i = 0; i < 4; i++)
			{
				if (i == LEFT) {
					m_center->nbrLFU[NE]->LF[i] = m_center->nbrLFU[N]->LF[RIGHT]; // LF 재사용
					m_center->nbrLFU[E]->LF[i] = m_center->LF[RIGHT];
					m_center->nbrLFU[SE]->LF[i] = m_center->nbrLFU[S]->LF[RIGHT];
				}
				else {
					m_center->nbrLFU[NE]->LF[i]->read_progress_odd = 0; // overwrite가 가능하도록 초기화
					m_center->nbrLFU[E]->LF[i]->read_progress_odd = 0;
					m_center->nbrLFU[SE]->LF[i]->read_progress_odd = 0;
				}
			}

			m_center->nbrLFU[NE]->id = m_center->nbrLFU[N]->id + 56; // id 업데이트
			m_center->nbrLFU[E]->id = m_center->id + 56;
			m_center->nbrLFU[SE]->id = m_center->nbrLFU[S]->id + 56;

			int f, r, b, l;
			find_LF_number_BMW(f, r, b, l, m_center->nbrLFU[NE]->id); // LF number 업데이트
			m_center->nbrLFU[NE]->LF[FRONT]->LF_number = f;
			m_center->nbrLFU[NE]->LF[RIGHT]->LF_number = r;
			m_center->nbrLFU[NE]->LF[BACK]->LF_number = b;
			m_center->nbrLFU[NE]->LF[LEFT]->LF_number = l;

			find_LF_number_BMW(f, r, b, l, m_center->nbrLFU[E]->id);
			m_center->nbrLFU[E]->LF[FRONT]->LF_number = f;
			m_center->nbrLFU[E]->LF[RIGHT]->LF_number = r;
			m_center->nbrLFU[E]->LF[BACK]->LF_number = b;
			m_center->nbrLFU[E]->LF[LEFT]->LF_number = l;

			find_LF_number_BMW(f, r, b, l, m_center->nbrLFU[SE]->id);
			m_center->nbrLFU[SE]->LF[FRONT]->LF_number = f;
			m_center->nbrLFU[SE]->LF[RIGHT]->LF_number = r;
			m_center->nbrLFU[SE]->LF[BACK]->LF_number = b;
			m_center->nbrLFU[SE]->LF[LEFT]->LF_number = l;

			// prev center's pinned even memory address goes to null while new center gets it
			m_center->LF[FRONT]->even_field = prevCenter->LF[FRONT]->even_field;
			m_center->LF[FRONT]->read_progress_even = 0;
			prevCenter->LF[FRONT]->even_field = nullptr;

			m_center->LF[RIGHT]->even_field = prevCenter->LF[LEFT]->even_field;
			m_center->LF[RIGHT]->read_progress_even = 0;
			prevCenter->LF[LEFT]->even_field = nullptr;

			m_center->LF[BACK]->even_field = prevCenter->LF[BACK]->even_field;
			m_center->LF[BACK]->read_progress_even = 0;
			prevCenter->LF[BACK]->even_field = nullptr;

			m_center->LF[LEFT]->even_field = prevCenter->LF[RIGHT]->even_field;
		} break;
		case -56: {
			// LF[LEFT] - nbr[NW], nbr[W], nbr[SW] should be replaced
			printf("[LFU_Window] Window Sliding - Left\n");
			m_center = prevCenter->nbrLFU[W];
			m_center->nbrLFU[N] = prevCenter->nbrLFU[NW]; // 새로운 LFU 관계 정의 (LFU Window Sliding)
			m_center->nbrLFU[NE] = prevCenter->nbrLFU[N];
			m_center->nbrLFU[E] = prevCenter;
			m_center->nbrLFU[SE] = prevCenter->nbrLFU[S];
			m_center->nbrLFU[S] = prevCenter->nbrLFU[SW];
			m_center->nbrLFU[SW] = prevCenter->nbrLFU[SE];
			m_center->nbrLFU[W] = prevCenter->nbrLFU[E];
			m_center->nbrLFU[NW] = prevCenter->nbrLFU[NE];

			std::swap(m_center->nbrLFU[NW]->LF[RIGHT], m_center->nbrLFU[NW]->LF[LEFT]);
			std::swap(m_center->nbrLFU[W]->LF[RIGHT], m_center->nbrLFU[W]->LF[LEFT]);
			std::swap(m_center->nbrLFU[SW]->LF[RIGHT], m_center->nbrLFU[SW]->LF[LEFT]);
			for (int i = 0; i < 4; i++)
			{
				if (i == RIGHT) {
					m_center->nbrLFU[NW]->LF[i] = m_center->nbrLFU[N]->LF[LEFT]; // LF 재사용
					m_center->nbrLFU[W]->LF[i] = m_center->LF[LEFT];
					m_center->nbrLFU[SW]->LF[i] = m_center->nbrLFU[S]->LF[LEFT];
				}
				else {
					m_center->nbrLFU[NW]->LF[i]->read_progress_odd = 0; // overwrite가 가능하도록 초기화
					m_center->nbrLFU[W]->LF[i]->read_progress_odd = 0;
					m_center->nbrLFU[SW]->LF[i]->read_progress_odd = 0;
				}
			}

			m_center->nbrLFU[NW]->LF[RIGHT] = m_center->nbrLFU[N]->LF[LEFT]; // LF 재사용
			m_center->nbrLFU[W]->LF[RIGHT] = m_center->LF[LEFT];
			m_center->nbrLFU[SW]->LF[RIGHT] = m_center->nbrLFU[S]->LF[LEFT];

			m_center->nbrLFU[NW]->id = m_center->nbrLFU[N]->id - 56;
			m_center->nbrLFU[W]->id = m_center->id - 56;
			m_center->nbrLFU[SW]->id = m_center->nbrLFU[S]->id - 56; // id 업데이트

			int f, r, b, l;
			find_LF_number_BMW(f, r, b, l, m_center->nbrLFU[SW]->id); // LF number 업데이트
			m_center->nbrLFU[SW]->LF[FRONT]->LF_number = f;
			m_center->nbrLFU[SW]->LF[RIGHT]->LF_number = r;
			m_center->nbrLFU[SW]->LF[BACK]->LF_number = b;
			m_center->nbrLFU[SW]->LF[LEFT]->LF_number = l;

			find_LF_number_BMW(f, r, b, l, m_center->nbrLFU[W]->id);
			m_center->nbrLFU[W]->LF[FRONT]->LF_number = f;
			m_center->nbrLFU[W]->LF[RIGHT]->LF_number = r;
			m_center->nbrLFU[W]->LF[BACK]->LF_number = b;
			m_center->nbrLFU[W]->LF[LEFT]->LF_number = l;

			find_LF_number_BMW(f, r, b, l, m_center->nbrLFU[NW]->id);
			m_center->nbrLFU[NW]->LF[FRONT]->LF_number = f;
			m_center->nbrLFU[NW]->LF[RIGHT]->LF_number = r;
			m_center->nbrLFU[NW]->LF[BACK]->LF_number = b;
			m_center->nbrLFU[NW]->LF[LEFT]->LF_number = l;

			// prev center's pinned even memory address goes to null while new center gets it
			m_center->LF[FRONT]->even_field = prevCenter->LF[FRONT]->even_field;
			m_center->LF[FRONT]->read_progress_even = 0;
			prevCenter->LF[FRONT]->even_field = nullptr;

			m_center->LF[LEFT]->even_field = prevCenter->LF[RIGHT]->even_field;
			m_center->LF[LEFT]->read_progress_even = 0;
			prevCenter->LF[RIGHT]->even_field = nullptr;

			m_center->LF[BACK]->even_field = prevCenter->LF[BACK]->even_field;
			m_center->LF[BACK]->read_progress_even = 0;
			prevCenter->LF[BACK]->even_field = nullptr;

			m_center->LF[RIGHT]->even_field = prevCenter->LF[LEFT]->even_field;
		} break;
		case 1: {
			printf("[LFU_Window] Window Sliding - Up\n");
			m_center = prevCenter->nbrLFU[N];
			m_center->nbrLFU[N] = prevCenter->nbrLFU[S];
			m_center->nbrLFU[NE] = prevCenter->nbrLFU[SE];
			m_center->nbrLFU[E] = prevCenter->nbrLFU[NE];
			m_center->nbrLFU[SE] = prevCenter->nbrLFU[E];
			m_center->nbrLFU[S] = prevCenter;
			m_center->nbrLFU[SW] = prevCenter->nbrLFU[W];
			m_center->nbrLFU[W] = prevCenter->nbrLFU[NW];
			m_center->nbrLFU[NW] = prevCenter->nbrLFU[SW];

			std::swap(m_center->nbrLFU[NE]->LF[FRONT], m_center->nbrLFU[NE]->LF[BACK]);
			std::swap(m_center->nbrLFU[N]->LF[FRONT], m_center->nbrLFU[N]->LF[BACK]);
			std::swap(m_center->nbrLFU[NW]->LF[FRONT], m_center->nbrLFU[NW]->LF[BACK]);
			for (int i = 0; i < 4; i++)
			{
				if (i == BACK) {
					m_center->nbrLFU[NE]->LF[i] = m_center->nbrLFU[E]->LF[FRONT]; // LF 재사용
					m_center->nbrLFU[N]->LF[i] = m_center->LF[FRONT];
					m_center->nbrLFU[NW]->LF[i] = m_center->nbrLFU[W]->LF[FRONT];
				}
				else {
					m_center->nbrLFU[NE]->LF[i]->read_progress_odd = 0; // overwrite가 가능하도록 초기화
					m_center->nbrLFU[N]->LF[i]->read_progress_odd = 0;
					m_center->nbrLFU[NW]->LF[i]->read_progress_odd = 0;
				}
			}

			m_center->nbrLFU[NE]->id = m_center->nbrLFU[E]->id + 1;
			m_center->nbrLFU[N]->id = m_center->id + 1;
			m_center->nbrLFU[NW]->id = m_center->nbrLFU[W]->id + 1;

			int f, r, b, l;
			find_LF_number_BMW(f, r, b, l, m_center->nbrLFU[N]->id); // LF number 업데이트
			m_center->nbrLFU[N]->LF[FRONT]->LF_number = f;
			m_center->nbrLFU[N]->LF[RIGHT]->LF_number = r;
			m_center->nbrLFU[N]->LF[BACK]->LF_number = b;
			m_center->nbrLFU[N]->LF[LEFT]->LF_number = l;

			find_LF_number_BMW(f, r, b, l, m_center->nbrLFU[NE]->id);
			m_center->nbrLFU[NE]->LF[FRONT]->LF_number = f;
			m_center->nbrLFU[NE]->LF[RIGHT]->LF_number = r;
			m_center->nbrLFU[NE]->LF[BACK]->LF_number = b;
			m_center->nbrLFU[NE]->LF[LEFT]->LF_number = l;

			find_LF_number_BMW(f, r, b, l, m_center->nbrLFU[NW]->id);
			m_center->nbrLFU[NW]->LF[FRONT]->LF_number = f;
			m_center->nbrLFU[NW]->LF[RIGHT]->LF_number = r;
			m_center->nbrLFU[NW]->LF[BACK]->LF_number = b;
			m_center->nbrLFU[NW]->LF[LEFT]->LF_number = l;

			// prev center's pinned even memory address goes to null while new center gets it
			m_center->LF[FRONT]->even_field = prevCenter->LF[BACK]->even_field;
			m_center->LF[FRONT]->read_progress_even = 0;
			prevCenter->LF[BACK]->even_field = nullptr;

			m_center->LF[RIGHT]->even_field = prevCenter->LF[RIGHT]->even_field;
			m_center->LF[RIGHT]->read_progress_even = 0;
			prevCenter->LF[RIGHT]->even_field = nullptr;

			m_center->LF[LEFT]->even_field = prevCenter->LF[LEFT]->even_field;
			m_center->LF[LEFT]->read_progress_even = 0;
			prevCenter->LF[LEFT]->even_field = nullptr;

			m_center->LF[BACK]->even_field = prevCenter->LF[FRONT]->even_field;
		} break;
		case -1: {
			printf("[LFU_Window] Window Sliding - Down\n");
			m_center = prevCenter->nbrLFU[S];
			m_center->nbrLFU[N] = prevCenter;
			m_center->nbrLFU[NE] = prevCenter->nbrLFU[E];
			m_center->nbrLFU[E] = prevCenter->nbrLFU[SE];
			m_center->nbrLFU[SE] = prevCenter->nbrLFU[NE];
			m_center->nbrLFU[S] = prevCenter->nbrLFU[N];
			m_center->nbrLFU[SW] = prevCenter->nbrLFU[NW];
			m_center->nbrLFU[W] = prevCenter->nbrLFU[SW];
			m_center->nbrLFU[NW] = prevCenter->nbrLFU[W];

			std::swap(m_center->nbrLFU[SE]->LF[FRONT], m_center->nbrLFU[SE]->LF[BACK]);
			std::swap(m_center->nbrLFU[S]->LF[FRONT], m_center->nbrLFU[S]->LF[BACK]);
			std::swap(m_center->nbrLFU[SW]->LF[FRONT], m_center->nbrLFU[SW]->LF[BACK]);
			for (int i = 0; i < 4; i++)
			{
				if (i == FRONT) {
					m_center->nbrLFU[SE]->LF[i] = m_center->nbrLFU[E]->LF[BACK]; // LF 재사용
					m_center->nbrLFU[S]->LF[i] = m_center->LF[BACK];
					m_center->nbrLFU[SW]->LF[i] = m_center->nbrLFU[W]->LF[BACK];
				}
				else {
					m_center->nbrLFU[SE]->LF[i]->read_progress_odd = 0; // overwrite가 가능하도록 초기화
					m_center->nbrLFU[S]->LF[i]->read_progress_odd = 0;
					m_center->nbrLFU[SW]->LF[i]->read_progress_odd = 0;
				}
			}

			m_center->nbrLFU[NE]->id = m_center->nbrLFU[E]->id + 1;
			m_center->nbrLFU[N]->id = m_center->id + 1;
			m_center->nbrLFU[NW]->id = m_center->nbrLFU[W]->id + 1;

			m_center->nbrLFU[SW]->id = m_center->nbrLFU[W]->id - 1;
			m_center->nbrLFU[S]->id = m_center->id - 1;
			m_center->nbrLFU[SE]->id = m_center->nbrLFU[E]->id - 1;

			int f, r, b, l;
			find_LF_number_BMW(f, r, b, l, m_center->nbrLFU[SW]->id); // LF number 업데이트
			m_center->nbrLFU[SW]->LF[FRONT]->LF_number = f;
			m_center->nbrLFU[SW]->LF[RIGHT]->LF_number = r;
			m_center->nbrLFU[SW]->LF[BACK]->LF_number = b;
			m_center->nbrLFU[SW]->LF[LEFT]->LF_number = l;

			find_LF_number_BMW(f, r, b, l, m_center->nbrLFU[S]->id);
			m_center->nbrLFU[S]->LF[FRONT]->LF_number = f;
			m_center->nbrLFU[S]->LF[RIGHT]->LF_number = r;
			m_center->nbrLFU[S]->LF[BACK]->LF_number = b;
			m_center->nbrLFU[S]->LF[LEFT]->LF_number = l;

			find_LF_number_BMW(f, r, b, l, m_center->nbrLFU[SE]->id);
			m_center->nbrLFU[SE]->LF[FRONT]->LF_number = f;
			m_center->nbrLFU[SE]->LF[RIGHT]->LF_number = r;
			m_center->nbrLFU[SE]->LF[BACK]->LF_number = b;
			m_center->nbrLFU[SE]->LF[LEFT]->LF_number = l;

			// prev center's pinned even memory address goes to null while new center gets it
			m_center->LF[BACK]->even_field = prevCenter->LF[FRONT]->even_field;
			m_center->LF[BACK]->read_progress_even = 0;
			prevCenter->LF[FRONT]->even_field = nullptr;

			m_center->LF[RIGHT]->even_field = prevCenter->LF[RIGHT]->even_field;
			m_center->LF[RIGHT]->read_progress_even = 0;
			prevCenter->LF[RIGHT]->even_field = nullptr;

			m_center->LF[LEFT]->even_field = prevCenter->LF[LEFT]->even_field;
			m_center->LF[LEFT]->read_progress_even = 0;
			prevCenter->LF[LEFT]->even_field = nullptr;

			m_center->LF[FRONT]->even_field = prevCenter->LF[BACK]->even_field;
		} break;
		}
		printf("Preparing new LFU window...\n");
	}
	// black lines (center)
	for (int dir = 0; dir < 4; dir++) {
		if (!isLFReadCompleted(m_center->LF[dir], ODD) && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_LF(m_center->LF[dir], BMW_LF[m_center->id][dir], ODD, curPosX, curPosY) < 0) return -1;
		}
	}
	*state_disk_read_thread = DISK_READ_THREAD_CENTER_LFU_ODD_FIELD_READ_COMPLETE;

	for (int dir = 0; dir < 4; dir++) {
		if (!isLFReadCompleted(m_center->LF[dir], EVEN) && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_LF(m_center->LF[dir], BMW_LF[m_center->id][dir], EVEN, curPosX, curPosY) < 0) return -1;
		}
	}
	*state_disk_read_thread = DISK_READ_THREAD_CENTER_LFU_EVEN_FIELD_READ_COMPLETE;

	set_LFU_read_priority(curPosX, curPosY);
	for (int i = 0; i < 8; i++)
	{
		LFU_NEIGHBOR nbrIdx = priority_LFU_read[i].second;
		read_LFs_with_priority(nbrIdx, curPosX, curPosY, main_thread_state);
	}
	*state_disk_read_thread = DISK_READ_THREAD_NEIGHBOR_LFU_READ_COMPLETE;
	
	return 0;
}

int LFU_Window::read_LF(Interlaced_LF* LF, std::string filename, const INTERLACE_FIELD& field, const int& curPosX, const int& curPosY)
{
	uint8_t* buf;
	size_t* progress;

	filename = LF_prefix + filename;
	if (field == ODD) {
		filename += "_odd.bgr";
		buf = LF->odd_field;
		progress = &LF->read_progress_odd;
	}
	else {
		filename += "_even.bgr";
		buf = LF->even_field;
		progress = &LF->read_progress_even;
	}

	FILE* fp = fopen(filename.c_str(), "rb");
	if (fp == nullptr) {
		printf("open failed, %s\n", filename.c_str());
		exit(1);
	}

	const size_t num_chunk = 50;
	const size_t chunk_size = this->interlaced_LF_size / num_chunk;
	size_t next_chunk_begin = *progress;
	for (int i = 0; i < num_chunk; i++) {
		if (curLFUID != getLFUID(curPosX, curPosY)) {
			*state_disk_read_thread = DISK_READ_THREAD_CENTER_LFU_READING;
			printf("LF read break\n");
			return -1; // Interrupt
		}
		*progress += fread(buf + next_chunk_begin, 1, sizeof(uint8_t) * chunk_size, fp);
		next_chunk_begin += chunk_size;
	}
	fclose(fp);

	return 0;
}

void LFU_Window::set_LFU_read_priority(const int& posX, const int& posY) 
{
	int lower_bound_x = (posX / 100) * 100;
	int lower_bound_y = (posY / 100) * 100;
	int upper_bound_x = lower_bound_x + 99;
	int upper_bound_y = lower_bound_y + 99;

	this->priority_LFU_read[0] = std::make_pair(getEuclideanDist(posX, posY, posX, upper_bound_y), N);
	this->priority_LFU_read[1] = std::make_pair(getEuclideanDist(posX, posY, upper_bound_x, upper_bound_y), NE);
	this->priority_LFU_read[2] = std::make_pair(getEuclideanDist(posX, posY, upper_bound_x, posY), E);
	this->priority_LFU_read[3] = std::make_pair(getEuclideanDist(posX, posY, upper_bound_x, lower_bound_y), SE);
	this->priority_LFU_read[4] = std::make_pair(getEuclideanDist(posX, posY, posX, lower_bound_y), S);
	this->priority_LFU_read[5] = std::make_pair(getEuclideanDist(posX, posY, lower_bound_x, lower_bound_y), SW);
	this->priority_LFU_read[6] = std::make_pair(getEuclideanDist(posX, posY, lower_bound_x, posY), W);
	this->priority_LFU_read[7] = std::make_pair(getEuclideanDist(posX, posY, lower_bound_x, upper_bound_y), NW);

	std::sort(priority_LFU_read.begin(), priority_LFU_read.end());
}

int LFU_Window::read_LFs_with_priority(LFU_NEIGHBOR nbrIdx, const int& curPosX, const int& curPosY, const MAIN_THREAD_STATE& main_thread_state)
{
	switch (nbrIdx)
	{
	case N: {
		if (!isLFReadCompleted(m_center->nbrLFU[nbrIdx]->LF[LEFT], ODD) && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_LF(m_center->nbrLFU[nbrIdx]->LF[LEFT], BMW_LF[m_center->nbrLFU[nbrIdx]->id][LEFT], ODD, curPosX, curPosY) < 0) return -1;
		}
		if (!isLFReadCompleted(m_center->nbrLFU[nbrIdx]->LF[RIGHT], ODD) && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_LF(m_center->nbrLFU[nbrIdx]->LF[RIGHT], BMW_LF[m_center->nbrLFU[nbrIdx]->id][RIGHT], ODD, curPosX, curPosY) < 0) return -1;
		}
		if (!isLFReadCompleted(m_center->nbrLFU[nbrIdx]->LF[FRONT], ODD) && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_LF(m_center->nbrLFU[nbrIdx]->LF[FRONT], BMW_LF[m_center->nbrLFU[nbrIdx]->id][FRONT], ODD, curPosX, curPosY) < 0) return -1;
		}
	} break;
	case NE: {
		if (!isLFReadCompleted(m_center->nbrLFU[nbrIdx]->LF[LEFT], ODD) && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_LF(m_center->nbrLFU[nbrIdx]->LF[LEFT], BMW_LF[m_center->nbrLFU[nbrIdx]->id][LEFT], ODD, curPosX, curPosY) < 0) return -1;
		}
		if (!isLFReadCompleted(m_center->nbrLFU[nbrIdx]->LF[BACK], ODD) && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_LF(m_center->nbrLFU[nbrIdx]->LF[BACK], BMW_LF[m_center->nbrLFU[nbrIdx]->id][BACK], ODD, curPosX, curPosY) < 0) return -1;
		}
		if (!isLFReadCompleted(m_center->nbrLFU[nbrIdx]->LF[FRONT], ODD) && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_LF(m_center->nbrLFU[nbrIdx]->LF[FRONT], BMW_LF[m_center->nbrLFU[nbrIdx]->id][FRONT], ODD, curPosX, curPosY) < 0) return -1;
		}
		if (!isLFReadCompleted(m_center->nbrLFU[nbrIdx]->LF[RIGHT], ODD) && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_LF(m_center->nbrLFU[nbrIdx]->LF[RIGHT], BMW_LF[m_center->nbrLFU[nbrIdx]->id][RIGHT], ODD, curPosX, curPosY) < 0) return -1;
		}
	} break;
	case E: {
		if (!isLFReadCompleted(m_center->nbrLFU[nbrIdx]->LF[FRONT], ODD) && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_LF(m_center->nbrLFU[nbrIdx]->LF[FRONT], BMW_LF[m_center->nbrLFU[nbrIdx]->id][FRONT], ODD, curPosX, curPosY) < 0) return -1;
		}
		if (!isLFReadCompleted(m_center->nbrLFU[nbrIdx]->LF[BACK], ODD) && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_LF(m_center->nbrLFU[nbrIdx]->LF[BACK], BMW_LF[m_center->nbrLFU[nbrIdx]->id][BACK], ODD, curPosX, curPosY) < 0) return -1;
		}
		if (!isLFReadCompleted(m_center->nbrLFU[nbrIdx]->LF[RIGHT], ODD) && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_LF(m_center->nbrLFU[nbrIdx]->LF[RIGHT], BMW_LF[m_center->nbrLFU[nbrIdx]->id][RIGHT], ODD, curPosX, curPosY) < 0) return -1;
		}
	} break;
	case SE: {
		if (!isLFReadCompleted(m_center->nbrLFU[nbrIdx]->LF[FRONT], ODD) && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_LF(m_center->nbrLFU[nbrIdx]->LF[FRONT], BMW_LF[m_center->nbrLFU[nbrIdx]->id][FRONT], ODD, curPosX, curPosY) < 0) return -1;
		}
		if (!isLFReadCompleted(m_center->nbrLFU[nbrIdx]->LF[LEFT], ODD) && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_LF(m_center->nbrLFU[nbrIdx]->LF[LEFT], BMW_LF[m_center->nbrLFU[nbrIdx]->id][LEFT], ODD, curPosX, curPosY) < 0) return -1;
		}
		if (!isLFReadCompleted(m_center->nbrLFU[nbrIdx]->LF[RIGHT], ODD) && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_LF(m_center->nbrLFU[nbrIdx]->LF[RIGHT], BMW_LF[m_center->nbrLFU[nbrIdx]->id][RIGHT], ODD, curPosX, curPosY) < 0) return -1;
		}
		if (!isLFReadCompleted(m_center->nbrLFU[nbrIdx]->LF[BACK], ODD) && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_LF(m_center->nbrLFU[nbrIdx]->LF[BACK], BMW_LF[m_center->nbrLFU[nbrIdx]->id][BACK], ODD, curPosX, curPosY) < 0) return -1;
		}
	} break;
	case S: {
		if (!isLFReadCompleted(m_center->nbrLFU[nbrIdx]->LF[LEFT], ODD) && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_LF(m_center->nbrLFU[nbrIdx]->LF[LEFT], BMW_LF[m_center->nbrLFU[nbrIdx]->id][LEFT], ODD, curPosX, curPosY) < 0) return -1;
		}
		if (!isLFReadCompleted(m_center->nbrLFU[nbrIdx]->LF[RIGHT], ODD) && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_LF(m_center->nbrLFU[nbrIdx]->LF[RIGHT], BMW_LF[m_center->nbrLFU[nbrIdx]->id][RIGHT], ODD, curPosX, curPosY) < 0) return -1;
		}
		if (!isLFReadCompleted(m_center->nbrLFU[nbrIdx]->LF[BACK], ODD) && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_LF(m_center->nbrLFU[nbrIdx]->LF[BACK], BMW_LF[m_center->nbrLFU[nbrIdx]->id][BACK], ODD, curPosX, curPosY) < 0) return -1;
		}
	} break;
	case SW: {
		if (!isLFReadCompleted(m_center->nbrLFU[nbrIdx]->LF[FRONT], ODD) && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_LF(m_center->nbrLFU[nbrIdx]->LF[FRONT], BMW_LF[m_center->nbrLFU[nbrIdx]->id][FRONT], ODD, curPosX, curPosY) < 0) return -1;
		}
		if (!isLFReadCompleted(m_center->nbrLFU[nbrIdx]->LF[RIGHT], ODD) && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_LF(m_center->nbrLFU[nbrIdx]->LF[RIGHT], BMW_LF[m_center->nbrLFU[nbrIdx]->id][RIGHT], ODD, curPosX, curPosY) < 0) return -1;
		}
		if (!isLFReadCompleted(m_center->nbrLFU[nbrIdx]->LF[BACK], ODD) && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_LF(m_center->nbrLFU[nbrIdx]->LF[BACK], BMW_LF[m_center->nbrLFU[nbrIdx]->id][BACK], ODD, curPosX, curPosY) < 0) return -1;
		}
		if (!isLFReadCompleted(m_center->nbrLFU[nbrIdx]->LF[LEFT], ODD) && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_LF(m_center->nbrLFU[nbrIdx]->LF[LEFT], BMW_LF[m_center->nbrLFU[nbrIdx]->id][LEFT], ODD, curPosX, curPosY) < 0) return -1;
		}
	} break;
	case W: {
		if (!isLFReadCompleted(m_center->nbrLFU[nbrIdx]->LF[FRONT], ODD) && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_LF(m_center->nbrLFU[nbrIdx]->LF[FRONT], BMW_LF[m_center->nbrLFU[nbrIdx]->id][FRONT], ODD, curPosX, curPosY) < 0) return -1;
		}
		if (!isLFReadCompleted(m_center->nbrLFU[nbrIdx]->LF[BACK], ODD) && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_LF(m_center->nbrLFU[nbrIdx]->LF[BACK], BMW_LF[m_center->nbrLFU[nbrIdx]->id][BACK], ODD, curPosX, curPosY) < 0) return -1;
		}
		if (!isLFReadCompleted(m_center->nbrLFU[nbrIdx]->LF[LEFT], ODD) && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_LF(m_center->nbrLFU[nbrIdx]->LF[LEFT], BMW_LF[m_center->nbrLFU[nbrIdx]->id][LEFT], ODD, curPosX, curPosY) < 0) return -1;
		}
	} break;
	case NW: {
		if (!isLFReadCompleted(m_center->nbrLFU[nbrIdx]->LF[RIGHT], ODD) && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_LF(m_center->nbrLFU[nbrIdx]->LF[RIGHT], BMW_LF[m_center->nbrLFU[nbrIdx]->id][RIGHT], ODD, curPosX, curPosY) < 0) return -1;
		}
		if (!isLFReadCompleted(m_center->nbrLFU[nbrIdx]->LF[BACK], ODD) && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_LF(m_center->nbrLFU[nbrIdx]->LF[BACK], BMW_LF[m_center->nbrLFU[nbrIdx]->id][BACK], ODD, curPosX, curPosY) < 0) return -1;
		}
		if (!isLFReadCompleted(m_center->nbrLFU[nbrIdx]->LF[LEFT], ODD) && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_LF(m_center->nbrLFU[nbrIdx]->LF[LEFT], BMW_LF[m_center->nbrLFU[nbrIdx]->id][LEFT], ODD, curPosX, curPosY) < 0) return -1;
		}
		if (!isLFReadCompleted(m_center->nbrLFU[nbrIdx]->LF[FRONT], ODD) && main_thread_state != MAIN_THREAD_TERMINATED) {
			if (read_LF(m_center->nbrLFU[nbrIdx]->LF[FRONT], BMW_LF[m_center->nbrLFU[nbrIdx]->id][FRONT], ODD, curPosX, curPosY) < 0) return -1;
		}
	} break;
	default: { 
		printf("INVALID neighbor\n");
		exit(1); }break;
	}

	return 0;
}

bool LFU_Window::isLFReadCompleted(Interlaced_LF* LF, const INTERLACE_FIELD& field)
{
	if (field == ODD)
	{
		if (LF->read_progress_odd == this->interlaced_LF_size)
			return true;
		else return false;
	}
	else
	{
		if (LF->read_progress_even == this->interlaced_LF_size)
			return true;
		else return false;
	}
}