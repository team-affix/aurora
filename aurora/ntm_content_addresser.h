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
			size_t memory_height = 0;
			size_t memory_width = 0;

		public:
			tensor key;
			tensor key_grad;
			tensor beta;
			tensor beta_grad;

		public:
			Sync internal_similarity;
			Ntm_sparsify internal_sparsify;
			Normalize internal_normalize;

		public:
			MODEL_FIELDS
			virtual ~ntm_content_addresser();
			ntm_content_addresser();
			ntm_content_addresser(size_t a_memory_height, size_t a_memory_width);

		};
		typedef ptr<ntm_content_addresser> Ntm_content_addresser;
	}
}
