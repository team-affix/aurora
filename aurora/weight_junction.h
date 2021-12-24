#pragma once
#include "affix-base/pch.h"
#include "model.h"
#include "weight_set.h"

namespace aurora {
	namespace models {
		class weight_junction : public model {
		public:
			size_t a;
			size_t b;
			vector<Weight_set> weight_sets;

		public:
			MODEL_FIELDS
			virtual ~weight_junction();
			weight_junction();
			weight_junction(size_t a_a, size_t a_b);

		};
		typedef affix_base::data::ptr<weight_junction> Weight_junction;
	}
}
