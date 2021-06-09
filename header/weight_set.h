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
			vector<ptr<weight>> weights;

		public:
			MODEL_FIELDS
			virtual ~weight_set();
			weight_set();
			weight_set(size_t a_a, function<void(ptr<param>&)> a_func);

		};
	}
}