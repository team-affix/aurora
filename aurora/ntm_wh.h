#pragma once
#include "pch.h"
#include "ntm_rh.h"

using aurora::models::model;

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
			ptr<ntm_rh> internal_rh;
			ptr<model> a_model;
			ptr<model> e_model;

		public:
			MODEL_FIELDS
			virtual ~ntm_wh();
			ntm_wh();
			ntm_wh(size_t a_units, vector<size_t> a_head_h_dims, size_t a_shift_units, function<void(ptr<param>&)> a_func);

		};
	}
}