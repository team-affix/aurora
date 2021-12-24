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
			aurora::maths::tensor key;
			aurora::maths::tensor key_grad;
			aurora::maths::tensor beta;
			aurora::maths::tensor beta_grad;
			aurora::maths::tensor wx;
			aurora::maths::tensor wx_grad;
			aurora::maths::tensor wy;
			aurora::maths::tensor wy_grad;
			aurora::maths::tensor g;
			aurora::maths::tensor g_grad;
			aurora::maths::tensor s;
			aurora::maths::tensor s_grad;
			aurora::maths::tensor gamma;
			aurora::maths::tensor gamma_grad;

		public:
			Ntm_content_addresser internal_content_addresser;
			Ntm_location_addresser internal_location_addresser;

		public:
			MODEL_FIELDS
			virtual ~ntm_addresser();
			ntm_addresser();
			ntm_addresser(size_t a_memory_height, size_t a_memory_width, std::vector<int> a_valid_shifts);

		};
		typedef affix_base::data::ptr<ntm_addresser> Ntm_addresser;
	}
}
