#pragma once
#include "pch.h"
#include "models.h"

using aurora::models::model;

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
			ptr<sync> internal_similarity;
			ptr<ntm_sparsify> internal_sparsify;
			ptr<normalize> internal_normalize;

		public:
			MODEL_FIELDS
			virtual ~ntm_content_addresser();
			ntm_content_addresser();
			ntm_content_addresser(size_t a_memory_height, size_t a_memory_width);

		};
	}
}
