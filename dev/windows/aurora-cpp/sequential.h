#pragma once
#include "model.h"
#include "param.h"
#include "param_sgd.h"
#include "param_mom.h"
#include <vector>
#include <initializer_list>

using aurora::modeling::model;
using std::vector;
using std::initializer_list;
using std::back_inserter;

namespace aurora {
	namespace modeling {
		class sequential : public model {
		public:
			vector<ptr<model>> models;

		public:
			virtual ~sequential();
			sequential();
			sequential(initializer_list<ptr<model>> a_il);

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