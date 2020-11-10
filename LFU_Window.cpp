#include "LFU_Window.h"

LFU_Window::LFU_Window(const int& posX, const int& posY, const size_t& light_field_size)
{
	printf("Allocating pinned memory");
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			printf(".");
			m_pinnedLFU[i][j] = alloc_uint8(light_field_size, "pinned");
		}
	}
	printf(" Completed\n");

	int LFUID = getLFUID(posX, posY);
	int LFUIDs[9];
	LFUIDs[N] = LFUID + 1;
	LFUIDs[NE] = LFUID + 5 + 1;
	LFUIDs[E] = LFUID + 5;
	LFUIDs[SE] = LFUID + 5 - 1;
	LFUIDs[S] = LFUID - 1;
	LFUIDs[SW] = LFUID - 5 - 1;
	LFUIDs[W] = LFUID - 5;
	LFUIDs[NW] = LFUID - 5 + 1;
	LFUIDs[8] = LFUID;
	construct_window(light_field_size);

	printf("Reading LFs");
	for (int i = 0; i < 9; i++)
	{
		printf(".");

		int f, r, b, l;
		find_LF_number_BMW(f, r, b, l, LFUIDs[i]);

		m_LFU[i].id = LFUIDs[i];
		if (m_LFU[i].LF[FRONT]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED) {
			read_uint8(m_LFU[i].LF[FRONT]->odd_field, BMW_LF[LFUIDs[i]][0], ODD);
			m_LFU[i].LF[FRONT]->LF_number = f;
			m_LFU[i].LF[FRONT]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
		}

		m_LFU[i].id = LFUIDs[i];
		if (m_LFU[i].LF[RIGHT]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED) {
			read_uint8(m_LFU[i].LF[RIGHT]->odd_field, BMW_LF[LFUIDs[i]][1], ODD);
			m_LFU[i].LF[RIGHT]->LF_number = r;
			m_LFU[i].LF[RIGHT]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
		}

		m_LFU[i].id = LFUIDs[i];
		if (m_LFU[i].LF[BACK]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED) {
			read_uint8(m_LFU[i].LF[BACK]->odd_field, BMW_LF[LFUIDs[i]][2], ODD);
			m_LFU[i].LF[BACK]->LF_number = b;
			m_LFU[i].LF[BACK]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
		} 
		m_LFU[i].id = LFUIDs[i];
		if (m_LFU[i].LF[LEFT]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED) {
			read_uint8(m_LFU[i].LF[LEFT]->odd_field, BMW_LF[LFUIDs[i]][3], ODD);
			m_LFU[i].LF[LEFT]->LF_number = l;
			m_LFU[i].LF[LEFT]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
		}
	}
	m_center = &m_LFU[8];

	read_uint8(m_center->LF[FRONT]->even_field, BMW_LF[m_center->id][0], EVEN);
	m_center->LF[FRONT]->progress = LF_READ_PROGRESS_EVEN_FIELD_PREPARED;

	read_uint8(m_center->LF[RIGHT]->even_field, BMW_LF[m_center->id][1], EVEN);
	m_center->LF[RIGHT]->progress = LF_READ_PROGRESS_EVEN_FIELD_PREPARED;

	read_uint8(m_center->LF[BACK]->even_field, BMW_LF[m_center->id][2], EVEN);
	m_center->LF[BACK]->progress = LF_READ_PROGRESS_EVEN_FIELD_PREPARED;

	read_uint8(m_center->LF[LEFT]->even_field, BMW_LF[m_center->id][3], EVEN);
	m_center->LF[LEFT]->progress = LF_READ_PROGRESS_EVEN_FIELD_PREPARED;

	memcpy(m_pinnedLFU[ODD][FRONT], m_center->LF[FRONT]->odd_field, light_field_size);
	memcpy(m_pinnedLFU[ODD][RIGHT], m_center->LF[RIGHT]->odd_field, light_field_size);
	memcpy(m_pinnedLFU[ODD][BACK], m_center->LF[BACK]->odd_field, light_field_size);
	memcpy(m_pinnedLFU[ODD][LEFT], m_center->LF[LEFT]->odd_field, light_field_size);
	memcpy(m_pinnedLFU[EVEN][FRONT], m_center->LF[FRONT]->even_field, light_field_size);
	memcpy(m_pinnedLFU[EVEN][RIGHT], m_center->LF[RIGHT]->even_field, light_field_size);
	memcpy(m_pinnedLFU[EVEN][BACK], m_center->LF[BACK]->even_field, light_field_size);
	memcpy(m_pinnedLFU[EVEN][LEFT], m_center->LF[LEFT]->even_field, light_field_size);
	pinned_memory_status = PINNED_LFU_EVEN_AVAILABLE;

	printf("Completed\n");

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
	for (int i = 0; i < 12; i++)
	{
		free_uint8(m_row[i].odd_field, "pageable");
		free_uint8(m_col[i].odd_field, "pageable");

		if (i == 4 || i == 7)
		{
			free_uint8(m_row[i].even_field, "pageable");
			free_uint8(m_col[i].even_field, "pageable");
		}
	}
	free_uint8(m_pinnedLFU[ODD][FRONT], "pinned");
	free_uint8(m_pinnedLFU[ODD][RIGHT], "pinned");
	free_uint8(m_pinnedLFU[ODD][BACK], "pinned");
	free_uint8(m_pinnedLFU[ODD][LEFT], "pinned");
	free_uint8(m_pinnedLFU[EVEN][FRONT], "pinned");
	free_uint8(m_pinnedLFU[EVEN][RIGHT], "pinned");
	free_uint8(m_pinnedLFU[EVEN][BACK], "pinned");
	free_uint8(m_pinnedLFU[EVEN][LEFT], "pinned");

}

void LFU_Window::construct_window(const size_t& light_field_size)
{
	for (int i = 0; i < 12; i++)
	{
		m_row[i].odd_field = alloc_uint8(light_field_size, "pageable");
		m_row[i].even_field = nullptr;
		m_row[i].type = ROW;

		m_col[i].odd_field = alloc_uint8(light_field_size, "pageable");
		m_col[i].even_field = nullptr;
		m_col[i].type = COL;

		if (i == 4 || i == 7) // m_center
		{
			m_row[i].even_field = alloc_uint8(light_field_size, "pageable");
			m_row[i].type = ROW;

			m_col[i].even_field = alloc_uint8(light_field_size, "pageable");
			m_col[i].type = COL;
		}
			
		m_row[i].progress = LF_READ_PROGRESS_NOT_PREPARED;
		m_col[i].progress = LF_READ_PROGRESS_NOT_PREPARED;
	}

	m_LFU[0].LF[FRONT] = &m_row[1];
	m_LFU[0].LF[RIGHT] = &m_col[6];
	m_LFU[0].LF[BACK] = &m_row[4];
	m_LFU[0].LF[LEFT] = &m_col[3];

	m_LFU[1].LF[FRONT] = &m_row[2];
	m_LFU[1].LF[RIGHT] = &m_col[9];
	m_LFU[1].LF[BACK] = &m_row[6];
	m_LFU[1].LF[LEFT] = &m_col[5];

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

void LFU_Window::update_window(const int& prevPosX, const int& prevPosY, const int& curPosX, const int& curPosY, const size_t& light_field_size, const MAIN_THREAD_STATE& main_thread_state)
{
	int prevLFUID = getLFUID(prevPosX, prevPosY);
	int curLFUID = getLFUID(curPosX, curPosY);

	switch (curLFUID - prevLFUID)
	{
	case 5: {
		// LF[RIGHT] - nbr[NE], nbr[E], nbr[SE] should be replaced
		pinned_memory_status = PINNED_LFU_NOT_AVAILABLE;

		m_center->nbr[NW]->LF[RIGHT] = m_center->nbr[NW]->LF[LEFT]; // 버려질 LF[LEFT]를 LF[RIGHT]에 저장하고
		m_center->nbr[W]->LF[RIGHT] = m_center->nbr[W]->LF[LEFT];
		m_center->nbr[SW]->LF[RIGHT] = m_center->nbr[SW]->LF[LEFT];
		m_center->nbr[NW]->LF[RIGHT]->progress = LF_READ_PROGRESS_NOT_PREPARED; // LF[RIGHT]가 덮어씌워질 수 있게 FLAG 마킹
		m_center->nbr[W]->LF[RIGHT]->progress = LF_READ_PROGRESS_NOT_PREPARED;
		m_center->nbr[SW]->LF[RIGHT]->progress = LF_READ_PROGRESS_NOT_PREPARED;
		m_center->nbr[NW]->LF[FRONT]->progress = LF_READ_PROGRESS_NOT_PREPARED;
		m_center->nbr[W]->LF[FRONT]->progress = LF_READ_PROGRESS_NOT_PREPARED;
		m_center->nbr[SW]->LF[FRONT]->progress = LF_READ_PROGRESS_NOT_PREPARED;
		m_center->nbr[SW]->LF[BACK]->progress = LF_READ_PROGRESS_NOT_PREPARED;
		
		m_center->nbr[NW]->LF[LEFT] = m_center->nbr[NE]->LF[RIGHT]; // LF[LEFT] 주소값 업데이트
		m_center->nbr[W]->LF[LEFT] = m_center->nbr[E]->LF[RIGHT];
		m_center->nbr[SW]->LF[LEFT] = m_center->nbr[SE]->LF[RIGHT];

		m_center->nbr[E]->nbr[N] = m_center->nbr[NE];
		m_center->nbr[E]->nbr[NE] = m_center->nbr[NW];
		m_center->nbr[E]->nbr[E] = m_center->nbr[W];
		m_center->nbr[E]->nbr[SE] = m_center->nbr[SW];
		m_center->nbr[E]->nbr[S] = m_center->nbr[SE];
		m_center->nbr[E]->nbr[SW] = m_center->nbr[S];
		m_center->nbr[E]->nbr[W] = m_center;
		m_center->nbr[E]->nbr[NW] = m_center->nbr[N];
		m_center = m_center->nbr[E];
		
		m_center->nbr[NE]->id = m_center->nbr[N]->id + 5; // id 업데이트
		m_center->nbr[E]->id = m_center->id + 5;
		m_center->nbr[SE]->id  = m_center->nbr[S]->id + 5;

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

	} break;
	case -5: {
		// LF[LEFT] - nbr[NW], nbr[W], nbr[SW] should be replaced
		pinned_memory_status = PINNED_LFU_NOT_AVAILABLE;

		m_center->nbr[NE]->LF[LEFT] = m_center->nbr[NE]->LF[RIGHT]; // 버려질 LF[RIGHT]를 LF[LEFT]에 저장하고
		m_center->nbr[E]->LF[LEFT] = m_center->nbr[E]->LF[RIGHT];
		m_center->nbr[SE]->LF[LEFT] = m_center->nbr[SE]->LF[RIGHT];
		m_center->nbr[NE]->LF[LEFT]->progress = LF_READ_PROGRESS_NOT_PREPARED; // LF[LEFT]가 덮어씌워질 수 있게 FLAG 마킹
		m_center->nbr[E]->LF[LEFT]->progress = LF_READ_PROGRESS_NOT_PREPARED;
		m_center->nbr[SE]->LF[LEFT]->progress = LF_READ_PROGRESS_NOT_PREPARED;
		m_center->nbr[NE]->LF[FRONT]->progress = LF_READ_PROGRESS_NOT_PREPARED;
		m_center->nbr[E]->LF[FRONT]->progress = LF_READ_PROGRESS_NOT_PREPARED;
		m_center->nbr[SE]->LF[FRONT]->progress = LF_READ_PROGRESS_NOT_PREPARED;
		m_center->nbr[SE]->LF[BACK]->progress = LF_READ_PROGRESS_NOT_PREPARED;

		m_center->nbr[NE]->LF[RIGHT] = m_center->nbr[NW]->LF[LEFT]; // LF[RIGHT] 주소값 업데이트
		m_center->nbr[E]->LF[RIGHT] = m_center->nbr[W]->LF[LEFT];
		m_center->nbr[SE]->LF[RIGHT] = m_center->nbr[SW]->LF[LEFT];

		m_center->nbr[W]->nbr[N] = m_center->nbr[NW]; // 새로운 LFU 관계 정의 (LFU Window Sliding)
		m_center->nbr[W]->nbr[NE] = m_center->nbr[N];
		m_center->nbr[W]->nbr[E] = m_center;
		m_center->nbr[W]->nbr[SE] = m_center->nbr[S];
		m_center->nbr[W]->nbr[S] = m_center->nbr[SW];
		m_center->nbr[W]->nbr[SW] = m_center->nbr[SE];
		m_center->nbr[W]->nbr[W] = m_center->nbr[E];
		m_center->nbr[W]->nbr[NW] = m_center->nbr[NE];
		m_center = m_center->nbr[W];

		m_center->nbr[SW]->id = m_center->nbr[S]->id - 5; // id 업데이트
		m_center->nbr[W]->id = m_center->id - 5;
		m_center->nbr[NW]->id = m_center->nbr[N]->id - 5; 

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
	} break;
	case 1: {
		// up - nbr[N], nbr[NE], nbr[NW] should be replaced
		pinned_memory_status = PINNED_LFU_NOT_AVAILABLE;

		m_center->nbr[SE]->LF[FRONT] = m_center->nbr[SE]->LF[BACK]; // 버려질 LF[BACK]를 LF[FRONT]에 저장하고
		m_center->nbr[S]->LF[FRONT] = m_center->nbr[S]->LF[BACK];
		m_center->nbr[SW]->LF[FRONT] = m_center->nbr[SW]->LF[BACK];
		m_center->nbr[SE]->LF[FRONT]->progress = LF_READ_PROGRESS_NOT_PREPARED; // LF[FRONT]가 덮어씌워질 수 있게 FLAG 마킹
		m_center->nbr[S]->LF[FRONT]->progress = LF_READ_PROGRESS_NOT_PREPARED;
		m_center->nbr[SW]->LF[FRONT]->progress = LF_READ_PROGRESS_NOT_PREPARED;
		m_center->nbr[SE]->LF[RIGHT]->progress = LF_READ_PROGRESS_NOT_PREPARED;
		m_center->nbr[S]->LF[RIGHT]->progress = LF_READ_PROGRESS_NOT_PREPARED;
		m_center->nbr[SW]->LF[RIGHT]->progress = LF_READ_PROGRESS_NOT_PREPARED;
		m_center->nbr[SW]->LF[LEFT]->progress = LF_READ_PROGRESS_NOT_PREPARED;

		m_center->nbr[SE]->LF[BACK] = m_center->nbr[NE]->LF[FRONT]; // LF[BACK] 주소값 업데이트
		m_center->nbr[S]->LF[BACK] = m_center->nbr[N]->LF[FRONT];
		m_center->nbr[SW]->LF[BACK] = m_center->nbr[NW]->LF[FRONT]; 

		m_center->nbr[N]->nbr[N] = m_center->nbr[S];
		m_center->nbr[N]->nbr[NE] = m_center->nbr[SE];
		m_center->nbr[N]->nbr[E] = m_center->nbr[NE];
		m_center->nbr[N]->nbr[SE] = m_center->nbr[E];
		m_center->nbr[N]->nbr[S] = m_center;
		m_center->nbr[N]->nbr[SW] = m_center->nbr[W];
		m_center->nbr[N]->nbr[W] = m_center->nbr[NW];
		m_center->nbr[N]->nbr[NW] = m_center->nbr[SW];
		m_center = m_center->nbr[N];

		m_center->nbr[N]->id = m_center->id + 1;
		m_center->nbr[NE]->id = m_center->nbr[E]->id + 1;
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
	} break;
	case -1: {
		// down - nbr[SE], nbr[S], nbr[SW] should be replaced
		pinned_memory_status = PINNED_LFU_NOT_AVAILABLE;

		m_center->nbr[NW]->LF[BACK] = m_center->nbr[NW]->LF[FRONT];
		m_center->nbr[N]->LF[BACK] = m_center->nbr[N]->LF[FRONT];
		m_center->nbr[NE]->LF[BACK] = m_center->nbr[NE]->LF[FRONT]; // 버려질 LF[FRONT]를 LF[BACK]에 저장하고
		m_center->nbr[NE]->LF[BACK]->progress = LF_READ_PROGRESS_NOT_PREPARED; // LF[BACK]가 덮어씌워질 수 있게 FLAG 마킹
		m_center->nbr[N]->LF[BACK]->progress = LF_READ_PROGRESS_NOT_PREPARED;
		m_center->nbr[NW]->LF[BACK]->progress = LF_READ_PROGRESS_NOT_PREPARED;
		m_center->nbr[NE]->LF[RIGHT]->progress = LF_READ_PROGRESS_NOT_PREPARED;
		m_center->nbr[N]->LF[RIGHT]->progress = LF_READ_PROGRESS_NOT_PREPARED;
		m_center->nbr[NW]->LF[RIGHT]->progress = LF_READ_PROGRESS_NOT_PREPARED;
		m_center->nbr[NW]->LF[LEFT]->progress = LF_READ_PROGRESS_NOT_PREPARED;

		m_center->nbr[NW]->LF[FRONT] = m_center->nbr[SW]->LF[BACK]; // LF[FRONT] 주소값 업데이트
		m_center->nbr[N]->LF[FRONT] = m_center->nbr[S]->LF[BACK];
		m_center->nbr[NE]->LF[FRONT] = m_center->nbr[SE]->LF[BACK];

		m_center->nbr[S]->nbr[N] = m_center;
		m_center->nbr[S]->nbr[NE] = m_center->nbr[E];
		m_center->nbr[S]->nbr[E] = m_center->nbr[SE];
		m_center->nbr[S]->nbr[SE] = m_center->nbr[NE];
		m_center->nbr[S]->nbr[S] = m_center->nbr[N];
		m_center->nbr[S]->nbr[SW] = m_center->nbr[NW];
		m_center->nbr[S]->nbr[W] = m_center->nbr[SW];
		m_center->nbr[S]->nbr[NW] = m_center->nbr[W];
		m_center = m_center->nbr[S];

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
	} break;
	}

	// black lines (center)

	if (m_center->LF[FRONT]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->LF[FRONT]->odd_field, BMW_LF[m_center->id][FRONT], ODD);
		m_center->LF[FRONT]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}
	if (m_center->LF[RIGHT]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->LF[RIGHT]->odd_field, BMW_LF[m_center->id][RIGHT], ODD);
		m_center->LF[RIGHT]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}
	if (m_center->LF[BACK]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->LF[BACK]->odd_field, BMW_LF[m_center->id][BACK], ODD);
		m_center->LF[BACK]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}
	if (m_center->LF[LEFT]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->LF[LEFT]->odd_field, BMW_LF[m_center->id][LEFT], ODD);
		m_center->LF[LEFT]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}
	memcpy(m_pinnedLFU[ODD][FRONT], m_center->LF[FRONT]->odd_field, light_field_size);
	memcpy(m_pinnedLFU[ODD][RIGHT], m_center->LF[RIGHT]->odd_field, light_field_size);
	memcpy(m_pinnedLFU[ODD][BACK], m_center->LF[BACK]->odd_field, light_field_size);
	memcpy(m_pinnedLFU[ODD][LEFT], m_center->LF[LEFT]->odd_field, light_field_size);
	pinned_memory_status = PINNED_LFU_ODD_AVAILABLE;

	if (m_center->LF[FRONT]->progress < LF_READ_PROGRESS_EVEN_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->LF[FRONT]->even_field, BMW_LF[m_center->id][FRONT], EVEN);
		m_center->LF[FRONT]->progress = LF_READ_PROGRESS_EVEN_FIELD_PREPARED;
	}
	if (m_center->LF[RIGHT]->progress < LF_READ_PROGRESS_EVEN_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->LF[RIGHT]->even_field, BMW_LF[m_center->id][RIGHT], EVEN);
		m_center->LF[RIGHT]->progress = LF_READ_PROGRESS_EVEN_FIELD_PREPARED;
	}
	if (m_center->LF[BACK]->progress < LF_READ_PROGRESS_EVEN_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->LF[BACK]->even_field, BMW_LF[m_center->id][BACK], EVEN);
		m_center->LF[BACK]->progress = LF_READ_PROGRESS_EVEN_FIELD_PREPARED;
	}
	if (m_center->LF[LEFT]->progress < LF_READ_PROGRESS_EVEN_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->LF[LEFT]->even_field, BMW_LF[m_center->id][LEFT], EVEN);
		m_center->LF[LEFT]->progress = LF_READ_PROGRESS_EVEN_FIELD_PREPARED;
	}
	memcpy(m_pinnedLFU[EVEN][FRONT], m_center->LF[FRONT]->even_field, light_field_size);
	memcpy(m_pinnedLFU[EVEN][RIGHT], m_center->LF[RIGHT]->even_field, light_field_size);
	memcpy(m_pinnedLFU[EVEN][BACK], m_center->LF[BACK]->even_field, light_field_size);
	memcpy(m_pinnedLFU[EVEN][LEFT], m_center->LF[LEFT]->even_field, light_field_size);
	pinned_memory_status = PINNED_LFU_EVEN_AVAILABLE;

	// green lines
	if(m_center->nbr[N]->LF[LEFT]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED){
		read_uint8(m_center->nbr[N]->LF[LEFT]->odd_field, BMW_LF[m_center->nbr[N]->id][LEFT], ODD);
		m_center->nbr[N]->LF[LEFT]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}
	if (m_center->nbr[N]->LF[RIGHT]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->nbr[N]->LF[RIGHT]->odd_field, BMW_LF[m_center->nbr[N]->id][RIGHT], ODD);
		m_center->nbr[N]->LF[RIGHT]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}
	if (m_center->nbr[S]->LF[LEFT]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->nbr[S]->LF[LEFT]->odd_field, BMW_LF[m_center->nbr[S]->id][LEFT], ODD);
		m_center->nbr[S]->LF[LEFT]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}
	if (m_center->nbr[S]->LF[RIGHT]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->nbr[S]->LF[RIGHT]->odd_field, BMW_LF[m_center->nbr[S]->id][RIGHT], ODD);
		m_center->nbr[S]->LF[RIGHT]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}
	if (m_center->nbr[E]->LF[FRONT]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->nbr[E]->LF[FRONT]->odd_field, BMW_LF[m_center->nbr[E]->id][FRONT], ODD);
		m_center->nbr[E]->LF[FRONT]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}
	if (m_center->nbr[E]->LF[BACK]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->nbr[E]->LF[BACK]->odd_field, BMW_LF[m_center->nbr[E]->id][BACK], ODD);
		m_center->nbr[E]->LF[BACK]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}
	if (m_center->nbr[W]->LF[FRONT]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->nbr[W]->LF[FRONT]->odd_field, BMW_LF[m_center->nbr[W]->id][FRONT], ODD);
		m_center->nbr[W]->LF[FRONT]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}
	if (m_center->nbr[W]->LF[BACK]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->nbr[W]->LF[BACK]->odd_field, BMW_LF[m_center->nbr[W]->id][BACK], ODD);
		m_center->nbr[W]->LF[BACK]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}

	// red lines
	if (m_center->nbr[N]->LF[FRONT]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->nbr[N]->LF[FRONT]->odd_field, BMW_LF[m_center->nbr[N]->id][FRONT], ODD);
		m_center->nbr[N]->LF[FRONT]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}
	if (m_center->nbr[E]->LF[RIGHT]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->nbr[E]->LF[RIGHT]->odd_field, BMW_LF[m_center->nbr[E]->id][RIGHT], ODD);
		m_center->nbr[E]->LF[RIGHT]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}
	if (m_center->nbr[S]->LF[BACK]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->nbr[S]->LF[BACK]->odd_field, BMW_LF[m_center->nbr[S]->id][BACK], ODD);
		m_center->nbr[S]->LF[BACK]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}
	if (m_center->nbr[W]->LF[LEFT]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->nbr[W]->LF[LEFT]->odd_field, BMW_LF[m_center->nbr[W]->id][LEFT], ODD);
		m_center->nbr[W]->LF[LEFT]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}

	// blue lines
	if (m_center->nbr[NE]->LF[FRONT]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->nbr[NE]->LF[FRONT]->odd_field, BMW_LF[m_center->nbr[NE]->id][FRONT], ODD);
		m_center->nbr[NE]->LF[FRONT]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}
	if (m_center->nbr[SE]->LF[BACK]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->nbr[SE]->LF[BACK]->odd_field, BMW_LF[m_center->nbr[SE]->id][BACK], ODD);
		m_center->nbr[SE]->LF[BACK]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}
	if (m_center->nbr[SW]->LF[BACK]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->nbr[SW]->LF[BACK]->odd_field, BMW_LF[m_center->nbr[SW]->id][BACK], ODD);
		m_center->nbr[SW]->LF[BACK]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}
	if (m_center->nbr[NW]->LF[FRONT]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->nbr[NW]->LF[FRONT]->odd_field, BMW_LF[m_center->nbr[NW]->id][FRONT], ODD);
		m_center->nbr[NW]->LF[FRONT]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}

	if (m_center->nbr[NE]->LF[RIGHT]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->nbr[NE]->LF[RIGHT]->odd_field, BMW_LF[m_center->nbr[NE]->id][RIGHT], ODD);
		m_center->nbr[NE]->LF[RIGHT]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}
	if (m_center->nbr[SE]->LF[RIGHT]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->nbr[SE]->LF[RIGHT]->odd_field, BMW_LF[m_center->nbr[SE]->id][RIGHT], ODD);
		m_center->nbr[SE]->LF[RIGHT]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}
	if (m_center->nbr[SW]->LF[LEFT]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->nbr[SW]->LF[LEFT]->odd_field, BMW_LF[m_center->nbr[SW]->id][LEFT], ODD);
		m_center->nbr[SW]->LF[LEFT]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}
	if (m_center->nbr[NW]->LF[LEFT]->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->nbr[NW]->LF[LEFT]->odd_field, BMW_LF[m_center->nbr[NW]->id][LEFT], ODD);
		m_center->nbr[NW]->LF[LEFT]->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}
}