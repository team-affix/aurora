#pragma once
#include "pch.h"
#include "model.h"

using aurora::models::model;

namespace aurora {
	namespace models {
		class ntm_rh : public model {
		public:
			size_t units = 0;
			size_t shift_units = 0;
			size_t y_units = 0;

		public:
			tensor key;
			tensor key_grad;
			tensor beta;
			tensor beta_grad;
			tensor g;
			tensor g_grad;
			tensor s;
			tensor s_grad;
			tensor gamma;
			tensor gamma_grad;

		public:
			ptr<model> key_model;
			ptr<model> beta_model;
			ptr<model> g_model;
			ptr<model> s_model;
			ptr<model> gamma_model;

		public:
			MODEL_FIELDS
			virtual ~ntm_rh();
			ntm_rh();
			ntm_rh(size_t a_units, vector<size_t> a_head_h_dims, size_t a_shift_units, function<void(ptr<param>&)> a_func);
		};
	}
}