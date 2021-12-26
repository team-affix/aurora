#pragma once
#include "affix-base/pch.h"
#include "model.h"

namespace aurora {
	namespace models {
		class ntm_rh : public model {
		public:
			size_t m_units = 0;
			size_t m_shift_units = 0;
			size_t m_y_units = 0;

		public:
			aurora::maths::tensor m_key;
			aurora::maths::tensor m_key_grad;
			aurora::maths::tensor m_beta;
			aurora::maths::tensor m_beta_grad;
			aurora::maths::tensor m_g;
			aurora::maths::tensor m_g_grad;
			aurora::maths::tensor m_s;
			aurora::maths::tensor m_s_grad;
			aurora::maths::tensor m_gamma;
			aurora::maths::tensor m_gamma_grad;

		public:
			Model m_key_model;
			Model m_beta_model;
			Model m_g_model;
			Model m_s_model;
			Model m_gamma_model;

		public:
			MODEL_FIELDS
			virtual ~ntm_rh();
			ntm_rh();
			ntm_rh(
				size_t a_units, 
				std::vector<size_t> a_head_h_dims, 
				size_t a_shift_units
			);

		};
		typedef affix_base::data::ptr<ntm_rh> Ntm_rh;
	}
}
