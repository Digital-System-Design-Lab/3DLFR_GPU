#include "LFU_Window.h"

LFU_Window::LFU_Window(const int& posX, const int& posY, const int& light_field_size)
{
	m_pinnedLFU[ODD][FRONT] = alloc_uint8(light_field_size, "pinned");
	m_pinnedLFU[ODD][RIGHT] = alloc_uint8(light_field_size, "pinned");
	m_pinnedLFU[ODD][BACK] = alloc_uint8(light_field_size, "pinned");
	m_pinnedLFU[ODD][LEFT] = alloc_uint8(light_field_size, "pinned");
	m_pinnedLFU[EVEN][FRONT] = alloc_uint8(light_field_size, "pinned");
	m_pinnedLFU[EVEN][RIGHT] = alloc_uint8(light_field_size, "pinned");
	m_pinnedLFU[EVEN][BACK] = alloc_uint8(light_field_size, "pinned");
	m_pinnedLFU[EVEN][LEFT] = alloc_uint8(light_field_size, "pinned");

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

	for (int i = 0; i < 9; i++)
	{
		printf("%d\n", i);

		m_LFU[i].id = LFUIDs[i];
		if (m_LFU[i].front->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED) {
			read_uint8(m_LFU[i].front->odd_field, BMW_LF[LFUIDs[i]][0], ODD);
			m_LFU[i].front->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
		}

		m_LFU[i].id = LFUIDs[i];
		if (m_LFU[i].right->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED) {
			read_uint8(m_LFU[i].right->odd_field, BMW_LF[LFUIDs[i]][1], ODD);
			m_LFU[i].right->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
		}

		m_LFU[i].id = LFUIDs[i];
		if (m_LFU[i].back->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED) {
			read_uint8(m_LFU[i].back->odd_field, BMW_LF[LFUIDs[i]][2], ODD);
			m_LFU[i].back->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
		} 
		m_LFU[i].id = LFUIDs[i];
		if (m_LFU[i].left->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED) {
			read_uint8(m_LFU[i].left->odd_field, BMW_LF[LFUIDs[i]][3], ODD);
			m_LFU[i].left->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
		}
	}
	m_center = &m_LFU[8];

	read_uint8(m_center->front->even_field, BMW_LF[m_center->id][0], EVEN);
	m_center->front->progress = LF_READ_PROGRESS_EVEN_FIELD_PREPARED;

	read_uint8(m_center->right->even_field, BMW_LF[m_center->id][1], EVEN);
	m_center->right->progress = LF_READ_PROGRESS_EVEN_FIELD_PREPARED;

	read_uint8(m_center->back->even_field, BMW_LF[m_center->id][2], EVEN);
	m_center->back->progress = LF_READ_PROGRESS_EVEN_FIELD_PREPARED;

	read_uint8(m_center->left->even_field, BMW_LF[m_center->id][3], EVEN);
	m_center->left->progress = LF_READ_PROGRESS_EVEN_FIELD_PREPARED;

	memcpy(m_pinnedLFU[ODD][FRONT], m_center->front->odd_field, light_field_size);
	memcpy(m_pinnedLFU[ODD][RIGHT], m_center->right->odd_field, light_field_size);
	memcpy(m_pinnedLFU[ODD][BACK], m_center->back->odd_field, light_field_size);
	memcpy(m_pinnedLFU[ODD][LEFT], m_center->left->odd_field, light_field_size);
	memcpy(m_pinnedLFU[EVEN][FRONT], m_center->front->even_field, light_field_size);
	memcpy(m_pinnedLFU[EVEN][RIGHT], m_center->right->even_field, light_field_size);
	memcpy(m_pinnedLFU[EVEN][BACK], m_center->back->even_field, light_field_size);
	memcpy(m_pinnedLFU[EVEN][LEFT], m_center->left->even_field, light_field_size);
	pinned_memory_status = PINNED_LFU_EVEN_AVAILABLE;

	printf("window read completed\n");

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

void LFU_Window::construct_window(const int& light_field_size)
{
	for (int i = 0; i < 12; i++)
	{
		printf("%d\n", i);

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

	m_LFU[0].front = &m_row[1];
	m_LFU[0].right = &m_col[6];
	m_LFU[0].back = &m_row[4];
	m_LFU[0].left = &m_col[3];

	m_LFU[1].front = &m_row[2];
	m_LFU[1].right = &m_col[9];
	m_LFU[1].back = &m_row[6];
	m_LFU[1].left = &m_col[5];

	m_LFU[2].front = &m_row[5];
	m_LFU[2].right = &m_col[10];
	m_LFU[2].back = &m_row[8];
	m_LFU[2].left = &m_col[7];

	m_LFU[3].front = &m_row[8];
	m_LFU[3].right = &m_col[11];
	m_LFU[3].back = &m_row[11];
	m_LFU[3].left = &m_col[8];

	m_LFU[4].front = &m_row[7];
	m_LFU[4].right = &m_col[8];
	m_LFU[4].back = &m_row[10];
	m_LFU[4].left = &m_col[5];

	m_LFU[5].front = &m_row[6];
	m_LFU[5].right = &m_col[5];
	m_LFU[5].back = &m_row[9];
	m_LFU[5].left = &m_col[2];

	m_LFU[6].front = &m_row[3];
	m_LFU[6].right = &m_col[4];
	m_LFU[6].back = &m_row[6];
	m_LFU[6].left = &m_col[1];

	m_LFU[7].front = &m_row[0];
	m_LFU[7].right = &m_col[3];
	m_LFU[7].back = &m_row[3];
	m_LFU[7].left = &m_col[0];

	m_LFU[8].front = &m_row[4]; // center
	m_LFU[8].right = &m_col[7];
	m_LFU[8].back = &m_row[7];
	m_LFU[8].left = &m_col[4];
}

void LFU_Window::update_window(const int& prevPosX, const int& prevPosY, const int& curPosX, const int& curPosY, const int& light_field_size, const MAIN_THREAD_STATE& main_thread_state)
{
	pinned_memory_status = PINNED_LFU_NOT_AVAILABLE;
	int prevLFUID = getLFUID(prevPosX, prevPosY);
	int curLFUID = getLFUID(curPosX, curPosY);

	switch (curLFUID - prevLFUID)
	{
	case 5: {
		// right - nbr[NE], nbr[E], nbr[SE] should be replaced
		m_center->nbr[NW]->right = m_center->nbr[NW]->left; // 버려질 left를 right에 저장하고
		m_center->nbr[W]->right = m_center->nbr[W]->left;
		m_center->nbr[SW]->right = m_center->nbr[SW]->left;
		m_center->nbr[NW]->right->progress = LF_READ_PROGRESS_NOT_PREPARED; // right가 덮어씌워질 수 있게 FLAG 마킹
		m_center->nbr[W]->right->progress = LF_READ_PROGRESS_NOT_PREPARED;
		m_center->nbr[SW]->right->progress = LF_READ_PROGRESS_NOT_PREPARED;
		m_center->nbr[NW]->front->progress = LF_READ_PROGRESS_NOT_PREPARED;
		m_center->nbr[W]->front->progress = LF_READ_PROGRESS_NOT_PREPARED;
		m_center->nbr[SW]->front->progress = LF_READ_PROGRESS_NOT_PREPARED;
		m_center->nbr[SW]->back->progress = LF_READ_PROGRESS_NOT_PREPARED;
		
		m_center->nbr[NW]->left = m_center->nbr[NE]->right; // left 주소값 업데이트
		m_center->nbr[W]->left = m_center->nbr[E]->right;
		m_center->nbr[SW]->left = m_center->nbr[SE]->right;

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
	} break;
	case -5: {
		// left - nbr[NW], nbr[W], nbr[SW] should be replaced
		m_center->nbr[NE]->left = m_center->nbr[NE]->right; // 버려질 right를 left에 저장하고
		m_center->nbr[E]->left = m_center->nbr[E]->right;
		m_center->nbr[SE]->left = m_center->nbr[SE]->right;
		m_center->nbr[NE]->left->progress = LF_READ_PROGRESS_NOT_PREPARED; // left가 덮어씌워질 수 있게 FLAG 마킹
		m_center->nbr[E]->left->progress = LF_READ_PROGRESS_NOT_PREPARED;
		m_center->nbr[SE]->left->progress = LF_READ_PROGRESS_NOT_PREPARED;
		m_center->nbr[NE]->front->progress = LF_READ_PROGRESS_NOT_PREPARED;
		m_center->nbr[E]->front->progress = LF_READ_PROGRESS_NOT_PREPARED;
		m_center->nbr[SE]->front->progress = LF_READ_PROGRESS_NOT_PREPARED;
		m_center->nbr[SE]->back->progress = LF_READ_PROGRESS_NOT_PREPARED;

		m_center->nbr[NE]->right = m_center->nbr[NW]->left; // right 주소값 업데이트
		m_center->nbr[E]->right = m_center->nbr[W]->left;
		m_center->nbr[SE]->right = m_center->nbr[SW]->left;

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
	} break;
	case 1: {
		// up - nbr[N], nbr[NE], nbr[NW] should be replaced
		m_center->nbr[SE]->front = m_center->nbr[SE]->back; // 버려질 back를 front에 저장하고
		m_center->nbr[S]->front = m_center->nbr[S]->back;
		m_center->nbr[SW]->front = m_center->nbr[SW]->back;
		m_center->nbr[SE]->front->progress = LF_READ_PROGRESS_NOT_PREPARED; // front가 덮어씌워질 수 있게 FLAG 마킹
		m_center->nbr[S]->front->progress = LF_READ_PROGRESS_NOT_PREPARED;
		m_center->nbr[SW]->front->progress = LF_READ_PROGRESS_NOT_PREPARED;
		m_center->nbr[SE]->right->progress = LF_READ_PROGRESS_NOT_PREPARED;
		m_center->nbr[S]->right->progress = LF_READ_PROGRESS_NOT_PREPARED;
		m_center->nbr[SW]->right->progress = LF_READ_PROGRESS_NOT_PREPARED;
		m_center->nbr[SW]->left->progress = LF_READ_PROGRESS_NOT_PREPARED;

		m_center->nbr[SE]->back = m_center->nbr[NE]->front; // back 주소값 업데이트
		m_center->nbr[S]->back = m_center->nbr[N]->front;
		m_center->nbr[SW]->back = m_center->nbr[NW]->front; 

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
	} break;
	case -1: {
		// down - nbr[SE], nbr[S], nbr[SW] should be replaced
		m_center->nbr[NW]->back = m_center->nbr[NW]->front;
		m_center->nbr[N]->back = m_center->nbr[N]->front;
		m_center->nbr[NE]->back = m_center->nbr[NE]->front; // 버려질 front를 back에 저장하고
		m_center->nbr[NE]->back->progress = LF_READ_PROGRESS_NOT_PREPARED; // back가 덮어씌워질 수 있게 FLAG 마킹
		m_center->nbr[N]->back->progress = LF_READ_PROGRESS_NOT_PREPARED;
		m_center->nbr[NW]->back->progress = LF_READ_PROGRESS_NOT_PREPARED;
		m_center->nbr[NE]->right->progress = LF_READ_PROGRESS_NOT_PREPARED;
		m_center->nbr[N]->right->progress = LF_READ_PROGRESS_NOT_PREPARED;
		m_center->nbr[NW]->right->progress = LF_READ_PROGRESS_NOT_PREPARED;
		m_center->nbr[NW]->left->progress = LF_READ_PROGRESS_NOT_PREPARED;

		m_center->nbr[NW]->front = m_center->nbr[SW]->back; // front 주소값 업데이트
		m_center->nbr[N]->front = m_center->nbr[S]->back;
		m_center->nbr[NE]->front = m_center->nbr[SE]->back;

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
	} break;
	}

	// black lines (center)

	if (m_center->front->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->front->odd_field, BMW_LF[m_center->id][FRONT], ODD);
		m_center->front->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}
	if (m_center->right->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->right->odd_field, BMW_LF[m_center->id][RIGHT], ODD);
		m_center->right->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}
	if (m_center->back->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->back->odd_field, BMW_LF[m_center->id][BACK], ODD);
		m_center->back->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}
	if (m_center->left->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->left->odd_field, BMW_LF[m_center->id][LEFT], ODD);
		m_center->left->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}
	memcpy(m_pinnedLFU[ODD][FRONT], m_center->front->odd_field, light_field_size);
	memcpy(m_pinnedLFU[ODD][RIGHT], m_center->right->odd_field, light_field_size);
	memcpy(m_pinnedLFU[ODD][BACK], m_center->back->odd_field, light_field_size);
	memcpy(m_pinnedLFU[ODD][LEFT], m_center->left->odd_field, light_field_size);
	pinned_memory_status = PINNED_LFU_ODD_AVAILABLE;

	if (m_center->front->progress < LF_READ_PROGRESS_EVEN_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->front->even_field, BMW_LF[m_center->id][FRONT], EVEN);
		m_center->front->progress = LF_READ_PROGRESS_EVEN_FIELD_PREPARED;
	}
	if (m_center->right->progress < LF_READ_PROGRESS_EVEN_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->right->even_field, BMW_LF[m_center->id][RIGHT], EVEN);
		m_center->right->progress = LF_READ_PROGRESS_EVEN_FIELD_PREPARED;
	}
	if (m_center->back->progress < LF_READ_PROGRESS_EVEN_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->back->even_field, BMW_LF[m_center->id][BACK], EVEN);
		m_center->back->progress = LF_READ_PROGRESS_EVEN_FIELD_PREPARED;
	}
	if (m_center->left->progress < LF_READ_PROGRESS_EVEN_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->left->even_field, BMW_LF[m_center->id][LEFT], EVEN);
		m_center->left->progress = LF_READ_PROGRESS_EVEN_FIELD_PREPARED;
	}
	memcpy(m_pinnedLFU[EVEN][FRONT], m_center->front->even_field, light_field_size);
	memcpy(m_pinnedLFU[EVEN][RIGHT], m_center->right->even_field, light_field_size);
	memcpy(m_pinnedLFU[EVEN][BACK], m_center->back->even_field, light_field_size);
	memcpy(m_pinnedLFU[EVEN][LEFT], m_center->left->even_field, light_field_size);
	pinned_memory_status = PINNED_LFU_EVEN_AVAILABLE;

	// green lines
	if(m_center->nbr[N]->left->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED){
		read_uint8(m_center->nbr[N]->left->odd_field, BMW_LF[m_center->nbr[N]->id][LEFT], ODD);
		m_center->nbr[N]->left->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}
	if (m_center->nbr[N]->right->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->nbr[N]->right->odd_field, BMW_LF[m_center->nbr[N]->id][RIGHT], ODD);
		m_center->nbr[N]->right->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}
	if (m_center->nbr[S]->left->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->nbr[S]->left->odd_field, BMW_LF[m_center->nbr[S]->id][LEFT], ODD);
		m_center->nbr[S]->left->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}
	if (m_center->nbr[S]->right->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->nbr[S]->right->odd_field, BMW_LF[m_center->nbr[S]->id][RIGHT], ODD);
		m_center->nbr[S]->right->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}
	if (m_center->nbr[E]->front->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->nbr[E]->front->odd_field, BMW_LF[m_center->nbr[E]->id][FRONT], ODD);
		m_center->nbr[E]->front->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}
	if (m_center->nbr[E]->back->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->nbr[E]->back->odd_field, BMW_LF[m_center->nbr[E]->id][BACK], ODD);
		m_center->nbr[E]->back->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}
	if (m_center->nbr[W]->front->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->nbr[W]->front->odd_field, BMW_LF[m_center->nbr[W]->id][FRONT], ODD);
		m_center->nbr[W]->front->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}
	if (m_center->nbr[W]->back->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->nbr[W]->back->odd_field, BMW_LF[m_center->nbr[W]->id][BACK], ODD);
		m_center->nbr[W]->back->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}

	// red lines
	if (m_center->nbr[N]->front->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->nbr[N]->front->odd_field, BMW_LF[m_center->nbr[N]->id][FRONT], ODD);
		m_center->nbr[N]->front->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}
	if (m_center->nbr[E]->right->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->nbr[E]->right->odd_field, BMW_LF[m_center->nbr[E]->id][RIGHT], ODD);
		m_center->nbr[E]->right->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}
	if (m_center->nbr[S]->back->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->nbr[S]->back->odd_field, BMW_LF[m_center->nbr[S]->id][BACK], ODD);
		m_center->nbr[S]->back->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}
	if (m_center->nbr[W]->left->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->nbr[W]->left->odd_field, BMW_LF[m_center->nbr[W]->id][LEFT], ODD);
		m_center->nbr[W]->left->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}

	// blue lines
	if (m_center->nbr[NE]->front->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->nbr[NE]->front->odd_field, BMW_LF[m_center->nbr[NE]->id][FRONT], ODD);
		m_center->nbr[NE]->front->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}
	if (m_center->nbr[SE]->back->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->nbr[SE]->back->odd_field, BMW_LF[m_center->nbr[SE]->id][BACK], ODD);
		m_center->nbr[SE]->back->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}
	if (m_center->nbr[SW]->back->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->nbr[SW]->back->odd_field, BMW_LF[m_center->nbr[SW]->id][BACK], ODD);
		m_center->nbr[SW]->back->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}
	if (m_center->nbr[NW]->front->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->nbr[NW]->front->odd_field, BMW_LF[m_center->nbr[NW]->id][FRONT], ODD);
		m_center->nbr[NW]->front->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}

	if (m_center->nbr[NE]->right->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->nbr[NE]->right->odd_field, BMW_LF[m_center->nbr[NE]->id][RIGHT], ODD);
		m_center->nbr[NE]->right->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}
	if (m_center->nbr[SE]->right->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->nbr[SE]->right->odd_field, BMW_LF[m_center->nbr[SE]->id][RIGHT], ODD);
		m_center->nbr[SE]->right->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}
	if (m_center->nbr[SW]->left->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->nbr[SW]->left->odd_field, BMW_LF[m_center->nbr[SW]->id][LEFT], ODD);
		m_center->nbr[SW]->left->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}
	if (m_center->nbr[NW]->left->progress < LF_READ_PROGRESS_ODD_FIELD_PREPARED && main_thread_state != MAIN_THREAD_TERMINATED) {
		read_uint8(m_center->nbr[NW]->left->odd_field, BMW_LF[m_center->nbr[NW]->id][LEFT], ODD);
		m_center->nbr[NW]->left->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}
}