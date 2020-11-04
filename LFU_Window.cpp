#include "LFU_Window.h"

LFU_Window::LFU_Window(const int& posX, const int& posY, const int& light_field_size)
{
	std::vector<int> LFUIDs = getLFUID(posX, posY);
	construct_window(light_field_size);

	for (int i = 0; i < LFUIDs.size(); i++)
	{
		int f, r, b, l;
		find_LF_number_BMW(f, r, b, l, LFUIDs[i]);
		m_LFU[i].front->LF_number = f;
		read_uint8(m_LFU[i].front->odd_field, BMW_LF[LFUIDs[i]][0], ODD);
		m_LFU[i].front->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;

		m_LFU[i].right->LF_number = r;
		read_uint8(m_LFU[i].right->odd_field, BMW_LF[LFUIDs[i]][1], ODD);
		m_LFU[i].right->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;

		m_LFU[i].back->LF_number = b;
		read_uint8(m_LFU[i].back->odd_field, BMW_LF[LFUIDs[i]][2], ODD);
		m_LFU[i].back->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;

		m_LFU[i].left->LF_number = l;
		read_uint8(m_LFU[i].left->odd_field, BMW_LF[LFUIDs[i]][3], ODD);
		m_LFU[i].left->progress = LF_READ_PROGRESS_ODD_FIELD_PREPARED;
	}
	m_center = &m_LFU[0];

	read_uint8(m_center->front->even_field, BMW_LF[LFUIDs[0]][0], EVEN);
	m_center->front->progress = LF_READ_PROGRESS_EVEN_FIELD_PREPARED;

	read_uint8(m_center->right->even_field, BMW_LF[LFUIDs[0]][1], EVEN);
	m_center->right->progress = LF_READ_PROGRESS_EVEN_FIELD_PREPARED;

	read_uint8(m_center->back->even_field, BMW_LF[LFUIDs[0]][2], EVEN);
	m_center->back->progress = LF_READ_PROGRESS_EVEN_FIELD_PREPARED;

	read_uint8(m_center->left->even_field, BMW_LF[LFUIDs[0]][3], EVEN);
	m_center->left->progress = LF_READ_PROGRESS_EVEN_FIELD_PREPARED;

	m_center->N = &m_LFU[8];
	m_center->N->S = m_center;
	m_center->NE = &m_LFU[1];
	m_center->NE->SW = m_center;
	m_center->E = &m_LFU[2];
	m_center->E->W = m_center;
	m_center->SE = &m_LFU[3];
	m_center->SE->NW = m_center;
	m_center->S = &m_LFU[4];
	m_center->S->N = m_center;
	m_center->SW = &m_LFU[5];
	m_center->SW->NE = m_center;
	m_center->W = &m_LFU[6];
	m_center->W->E = m_center;
	m_center->NW = &m_LFU[7];
}

LFU_Window::~LFU_Window()
{
	for (int i = 0; i < 12; i++)
	{
		free_uint8(m_row[i].odd_field, "pinned");
		free_uint8(m_col[i].odd_field, "pinned");

		if (i == 4 || i == 7)
		{
			free_uint8(m_row[i].even_field, "pinned");
			free_uint8(m_col[i].even_field, "pinned");
		}
	}
}

void LFU_Window::construct_window(const int& light_field_size)
{
	for (int i = 0; i < 12; i++)
	{
		m_row[i].LF_number = -1;
		m_row[i].odd_field = alloc_uint8(light_field_size, "pinned");
		m_row[i].even_field = nullptr;
		m_row[i].type = ROW;

		m_col[i].LF_number = -1;
		m_col[i].odd_field = alloc_uint8(light_field_size, "pinned");
		m_col[i].even_field = nullptr;
		m_col[i].type = COL;

		if (i == 4 || i == 7)
		{
			m_row[i].LF_number = -1;
			m_row[i].even_field = alloc_uint8(light_field_size, "pinned");

			m_col[i].LF_number = -1;
			m_col[i].even_field = alloc_uint8(light_field_size, "pinned");
		}
		m_row[i].progress = LF_READ_PROGRESS_NOT_PREPARED;
		m_col[i].progress = LF_READ_PROGRESS_NOT_PREPARED;
	}

	m_LFU[0].front = &m_row[4];
	m_LFU[0].right = &m_col[7];
	m_LFU[0].back = &m_row[7];
	m_LFU[0].left = &m_col[4];

	m_LFU[8].front = &m_row[1];
	m_LFU[8].right = &m_col[6];
	m_LFU[8].back = &m_row[4];
	m_LFU[8].left = &m_col[3];

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

	m_LFU[8].front = &m_row[0];
	m_LFU[8].right = &m_col[3];
	m_LFU[8].back = &m_row[3];
	m_LFU[8].left = &m_col[0];
}