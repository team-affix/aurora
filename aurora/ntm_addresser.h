#pragma once
#include "affix-base/pch.h"
#include "model.h"
#include "ntm_content_addresser.h"
#include "ntm_location_addresser.h"

namespace aurora {
	namespace models {
		class ntm_addresser : public model {
		public:
			size_t m_memory_height = 0;
			size_t m_memory_width = 0;
			size_t m_shift_units = 0;

		public:
			aurora::maths::tensor m_key;
			aurora::maths::tensor m_key_grad;
			aurora::maths::tensor m_beta;
			aurora::maths::tensor m_beta_grad;
			aurora::maths::tensor m_wx;
			aurora::maths::tensor m_wx_grad;
			aurora::maths::tensor m_wy;
			aurora::maths::tensor m_wy_grad;
			aurora::maths::tensor m_g;
			aurora::maths::tensor m_g_grad;
			aurora::maths::tensor m_s;
			aurora::maths::tensor m_s_grad;
			aurora::maths::tensor m_gamma;
			aurora::maths::tensor m_gamma_grad;

		public:
			Ntm_content_addresser m_internal_content_addresser;
			Ntm_location_addresser m_internal_location_addresser;

		public:
			MODEL_FIELDS
			virtual ~ntm_addresser();
			ntm_addresser();
			ntm_addresser(
				size_t a_memory_height, 
				size_t a_memory_width, 
				std::vector<int> a_valid_shifts
			);

		};
		typedef affix_base::data::ptr<ntm_addresser> Ntm_addresser;
	}
}
