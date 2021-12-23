#pragma once
#include "affix-base/pch.h"
#include "model.h"
#include "ntm_content_addresser.h"
#include "ntm_location_addresser.h"

namespace aurora {
	namespace models {
		class ntm_addresser : public model {
		public:
			size_t memory_height = 0;
			size_t memory_width = 0;
			size_t shift_units = 0;

		public:
			tensor key;
			tensor key_grad;
			tensor beta;
			tensor beta_grad;
			tensor wx;
			tensor wx_grad;
			tensor wy;
			tensor wy_grad;
			tensor g;
			tensor g_grad;
			tensor s;
			tensor s_grad;
			tensor gamma;
			tensor gamma_grad;

		public:
			Ntm_content_addresser internal_content_addresser;
			Ntm_location_addresser internal_location_addresser;

		public:
			MODEL_FIELDS
			virtual ~ntm_addresser();
			ntm_addresser();
			ntm_addresser(size_t a_memory_height, size_t a_memory_width, vector<int> a_valid_shifts);

		};
		typedef ptr<ntm_addresser> Ntm_addresser;
	}
}