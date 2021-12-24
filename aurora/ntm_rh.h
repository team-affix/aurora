#pragma once
#include "affix-base/pch.h"
#include "model.h"

namespace aurora {
	namespace models {
		class ntm_rh : public model {
		public:
			size_t units = 0;
			size_t shift_units = 0;
			size_t y_units = 0;

		public:
			aurora::maths::tensor key;
			aurora::maths::tensor key_grad;
			aurora::maths::tensor beta;
			aurora::maths::tensor beta_grad;
			aurora::maths::tensor g;
			aurora::maths::tensor g_grad;
			aurora::maths::tensor s;
			aurora::maths::tensor s_grad;
			aurora::maths::tensor gamma;
			aurora::maths::tensor gamma_grad;

		public:
			Model key_model;
			Model beta_model;
			Model g_model;
			Model s_model;
			Model gamma_model;

		public:
			MODEL_FIELDS
			virtual ~ntm_rh();
			ntm_rh();
			ntm_rh(size_t a_units, vector<size_t> a_head_h_dims, size_t a_shift_units);

		};
		typedef affix_base::data::ptr<ntm_rh> Ntm_rh;
	}
}
