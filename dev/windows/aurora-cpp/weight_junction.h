#pragma once
#include "model.h"
#include "weight_set.h"
#include <vector>

using aurora::modeling::model;
using aurora::modeling::weight_set;
using std::vector;

namespace aurora {
	namespace modeling {
		class weight_junction : public model {
		public:
			size_t a;
			size_t b;
			vector<ptr<weight_set>> weight_sets;

		public:
			MODEL_FIELDS
			virtual ~weight_junction();
			weight_junction();
			weight_junction(size_t a_a, size_t a_b, function<void(ptr<param>&)> a_init);

		};
	}
}