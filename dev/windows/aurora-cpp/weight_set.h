#pragma once
#include "model.h"
#include "weight.h"

using aurora::modeling::weight;

namespace aurora {
	namespace modeling {
		class weight_set : public model {
		public:
			size_t a;
			vector<ptr<weight>> weights;

		public:
			MODEL_FIELDS
			virtual ~weight_set();
			weight_set();
			weight_set(size_t a_a, function<void(ptr<param>&)> a_init);

		};
	}
}