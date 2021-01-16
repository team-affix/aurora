#pragma once
#include "model.h"
#include <vector>

using aurora::modeling::model;
using std::vector;

namespace aurora {
	namespace modeling {
		class layer : public model {
		public:
			vector<ptr<model>> models;

		public:
			virtual ~layer();
			layer();
			layer(size_t a_a, ptr<model> a_model_template, vector<param*>& a_pl);
			layer(size_t a_a, ptr<model> a_model_template, vector<param_sgd*>& a_pl);
			layer(size_t a_a, ptr<model> a_model_template, vector<param_mom*>& a_pl);
			layer(initializer_list<ptr<model>> a_il);

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