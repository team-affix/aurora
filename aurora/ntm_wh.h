#pragma once
#include "affix-base/pch.h"
#include "ntm_rh.h"

namespace aurora {
	namespace models {
		class ntm_wh : public model {
		public:
			size_t units = 0;

		public:
			aurora::maths::tensor a;
			aurora::maths::tensor a_grad;
			aurora::maths::tensor e;
			aurora::maths::tensor e_grad;

		public:
			Ntm_rh internal_rh;
			Model a_model;
			Model e_model;

		public:
			MODEL_FIELDS
			virtual ~ntm_wh();
			ntm_wh();
			ntm_wh(size_t a_units, std::vector<size_t> a_head_h_dims, size_t a_shift_units);

		};
		typedef affix_base::data::ptr<ntm_wh> Ntm_wh;
	}
}
