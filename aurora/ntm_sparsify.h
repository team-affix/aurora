#pragma once
#include "pch.h"
#include "model.h"

using aurora::models::model;

namespace aurora {
	namespace models {
		class ntm_sparsify : public model {
		public:
			size_t memory_height = 0;

		public:
			tensor beta = 0;
			tensor beta_grad = 0;

		public:
			MODEL_FIELDS
			virtual ~ntm_sparsify();
			ntm_sparsify();
			ntm_sparsify(size_t a_memory_height);

		};
		typedef ptr<ntm_sparsify> Ntm_sparsify;
	}
}