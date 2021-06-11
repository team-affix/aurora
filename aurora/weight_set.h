#pragma once
#include "pch.h"
#include "model.h"
#include "weight.h"

using aurora::models::weight;

namespace aurora {
	namespace models {
		class weight_set : public model {
		public:
			size_t a;
			vector<Weight> weights;

		public:
			MODEL_FIELDS
			virtual ~weight_set();
			weight_set();
			weight_set(size_t a_a, function<void(Param&)> a_func);

		};
		typedef ptr<weight_set> Weight_set;
	}
}