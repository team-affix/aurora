#pragma once
#include "model.h"
#include "weight.h"
#include "param.h"
#include "param_sgd.h"
#include "param_mom.h"
#include <vector>

using aurora::modeling::model;
using aurora::modeling::weight;
using std::vector;

namespace aurora {
	namespace modeling {
		class weight_set : public model {
		public:
			vector<weight> weights;

		public:
			virtual ~weight_set();
			weight_set();
			weight_set(size_t a, vector<param*>& pl);
			weight_set(size_t a, vector<param_sgd*>& pl);
			weight_set(size_t a, vector<param_mom*>& pl);

		public:
			virtual void fwd();
			virtual void bwd();
			virtual void recur(function<void(model&)> func);
			virtual void compile();

		};
	}
}