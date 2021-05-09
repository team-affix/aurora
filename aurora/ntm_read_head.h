#pragma once
#include "pch.h"
#include "model.h"
#include "layer.h"

using aurora::models::model;
using aurora::models::layer;

namespace aurora {
	namespace models {
		class ntm_read_head : public model {
		public:
			size_t units;
			size_t shift_units;
			size_t lr_units;
			size_t sm_units;

		public:
			// LR, SIZE == units
			tensor k;
			tensor k_grad;
			// LR, SIZE == 1
			tensor beta;
			tensor beta_grad;
			// SM, SIZE == 1
			tensor g;
			tensor g_grad;
			// SM, SIZE == shift_units
			tensor s;
			tensor s_grad;
			// LR, SIZE == 1
			tensor gamma;
			tensor gamma_grad;
			
		public:
			ptr<model> internal_model;
			// SIZE == units + 2
			ptr<layer> lr_layer;
			// SIZE == shift_units + 1
			ptr<layer> sm_layer;

		public:
			MODEL_FIELDS
			virtual ~ntm_read_head();
			ntm_read_head();
			ntm_read_head(vector<size_t> a_dims, size_t a_shift_units, function<void(ptr<param>&)> a_func);

		};
	}
}