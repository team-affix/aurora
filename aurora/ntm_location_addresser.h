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
			size_t memory_height = 0;
			size_t shift_units = 0;

		public:
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
			Interpolate internal_interpolate;
			Shift internal_shift;
			Power internal_power;
			Normalize internal_normalize;

		public:
			MODEL_FIELDS
			virtual ~ntm_location_addresser();
			ntm_location_addresser();
			ntm_location_addresser(size_t a_memory_height, vector<int> a_valid_shifts);

		};
		typedef affix_base::data::ptr<ntm_location_addresser> Ntm_location_addresser;
	}
}
