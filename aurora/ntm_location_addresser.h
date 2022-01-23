#pragma once
#include "affix-base/pch.h"
#include "model.h"
#include "interpolate.h"
#include "shift.h"
#include "power.h"
#include "normalize.h"

namespace aurora {
	namespace models {
		class ntm_location_addresser : public model {
		public:
			size_t m_memory_height = 0;
			size_t m_shift_units = 0;

		public:
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
			Interpolate m_internal_interpolate;
			Shift m_internal_shift;
			Power m_internal_power;
			Normalize m_internal_normalize;

		public:
			MODEL_FIELDS
			virtual ~ntm_location_addresser();
			ntm_location_addresser();
			ntm_location_addresser(
				size_t a_memory_height, 
				std::vector<int> a_valid_shifts
			);

		};
		typedef affix_base::data::ptr<ntm_location_addresser> Ntm_location_addresser;
	}
}
