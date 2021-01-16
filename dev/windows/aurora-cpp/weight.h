#pragma once
#include "model.h"
#include "param.h"

using aurora::modeling::model;

namespace aurora {
	namespace modeling {
		class weight : public model {
		public:
			ptr<param> pmt = new param();

		public:
			virtual ~weight();
			weight();
			weight(vector<param*>& a_pl);
			weight(vector<param_sgd*>& a_pl);
			weight(vector<param_mom*>& a_pl);

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