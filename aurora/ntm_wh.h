#pragma once
#include "affix-base/pch.h"
#include "ntm_rh.h"

namespace aurora {
	namespace models {
		class ntm_wh : public model {
		public:
			size_t units = 0;

		public:
			tensor a;
			tensor a_grad;
			tensor e;
			tensor e_grad;

		public:
			Ntm_rh internal_rh;
			Model a_model;
			Model e_model;

		public:
			MODEL_FIELDS
			virtual ~ntm_wh();
			ntm_wh();
			ntm_wh(size_t a_units, vector<size_t> a_head_h_dims, size_t a_shift_units);

		};
		typedef ptr<ntm_wh> Ntm_wh;
	}
}