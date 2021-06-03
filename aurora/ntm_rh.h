#pragma once
#include "pch.h"
#include "model.h"
#include "layer.h"
#include "normalize.h"

using aurora::models::model;
using aurora::models::layer;

namespace aurora {
	namespace models {
		class ntm_rh : public model {
		public:
			size_t units;
			size_t s_units;
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
			ptr<model> lr_model;
			ptr<model> sm_model;
			ptr<normalize> internal_shift_normalize;

		public:
			MODEL_FIELDS
			virtual ~ntm_rh();
			ntm_rh();
			ntm_rh(size_t a_units, vector<size_t> a_h_dims, size_t a_s_units, function<void(ptr<param>&)> a_func);

		};
	}
}
