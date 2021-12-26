#pragma once
#include "affix-base/pch.h"
#include "model.h"
#include "sync.h"
#include "cos_sim.h"
#include "ntm_sparsify.h"
#include "normalize.h"

namespace aurora {
	namespace models {
		class ntm_content_addresser : public model {
		public:
			size_t m_memory_height = 0;
			size_t m_memory_width = 0;

		public:
			aurora::maths::tensor m_key;
			aurora::maths::tensor m_key_grad;
			aurora::maths::tensor m_beta;
			aurora::maths::tensor m_beta_grad;

		public:
			Sync m_internal_similarity;
			Ntm_sparsify m_internal_sparsify;
			Normalize m_internal_normalize;

		public:
			MODEL_FIELDS
			virtual ~ntm_content_addresser();
			ntm_content_addresser();
			ntm_content_addresser(
				size_t a_memory_height, 
				size_t a_memory_width
			);

		};
		typedef affix_base::data::ptr<ntm_content_addresser> Ntm_content_addresser;
	}
}
