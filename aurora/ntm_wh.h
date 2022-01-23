#pragma once
#include "affix-base/pch.h"
#include "ntm_rh.h"

namespace aurora {
	namespace models {
		class ntm_wh : public model {
		public:
			size_t m_units = 0;

		public:
			aurora::maths::tensor m_a;
			aurora::maths::tensor m_a_grad;
			aurora::maths::tensor m_e;
			aurora::maths::tensor m_e_grad;

		public:
			Ntm_rh m_internal_rh;
			Model m_a_model;
			Model m_e_model;

		public:
			MODEL_FIELDS
			virtual ~ntm_wh();
			ntm_wh();
			ntm_wh(
				size_t a_units,
				std::vector<size_t> a_head_h_dims,
				size_t a_shift_units
			);

		};
		typedef affix_base::data::ptr<ntm_wh> Ntm_wh;
	}
}
