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
			virtual ~weight_junction();
			weight_junction();
			weight_junction(size_t a_a, size_t a_b, vector<param*>& a_pl);
			weight_junction(size_t a_a, size_t a_b, vector<param_sgd*>& a_pl);
			weight_junction(size_t a_a, size_t a_b, vector<param_mom*>& a_pl);

		public:
			virtual model* clone();
			virtual model* clone(vector<param*>& a_pl);
			virtual model* clone(vector<param_sgd*>& a_pl);
			virtual model* clone(vector<param_mom*>& a_pl);

		public:
			virtual void fwd();
			virtual void bwd();
			virtual tensor& fwd(tensor a_x);
			virtual tensor& bwd(tensor a_y_grad);
			virtual void signal(tensor a_y_des);

		public:
			virtual void cycle(tensor a_x, tensor a_y_des);

		public:
			virtual void recur(function<void(model*)> a_func);

		public:
			virtual void compile();

		};
	}
}