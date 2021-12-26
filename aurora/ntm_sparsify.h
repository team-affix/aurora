#pragma once
#include "affix-base/pch.h"
#include "model.h"

namespace aurora {
	namespace models {
		class ntm_sparsify : public model {
		public:
			size_t m_memory_height = 0;

		public:
			aurora::maths::tensor m_beta = 0;
			aurora::maths::tensor m_beta_grad = 0;

		public:
			MODEL_FIELDS
			virtual ~ntm_sparsify();
			ntm_sparsify();
			ntm_sparsify(
				size_t a_memory_height
			);

		};
		typedef affix_base::data::ptr<ntm_sparsify> Ntm_sparsify;
	}
}
