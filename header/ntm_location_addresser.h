#pragma once
#include "pch.h"
#include "model.h"
#include "interpolate.h"
#include "shift.h"
#include "power.h"
#include "normalize.h"

using aurora::models::model;
using aurora::models::interpolate;
using aurora::models::shift;
using aurora::models::power;
using aurora::models::normalize;

namespace aurora {
	namespace models {
		class ntm_location_addresser : public model {
		public:
			size_t memory_height = 0;
			size_t shift_units = 0;

		public:
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
			ptr<interpolate> internal_interpolate;
			ptr<shift> internal_shift;
			ptr<power> internal_power;
			ptr<normalize> internal_normalize;

		public:
			MODEL_FIELDS
			virtual ~ntm_location_addresser();
			ntm_location_addresser();
			ntm_location_addresser(size_t a_memory_height, vector<int> a_valid_shifts);

		};
	}
}