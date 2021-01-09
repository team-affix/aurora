#pragma once
#include "ptr.h"
#include "ref.h"
#include "tensor.h"
#include "param.h"
#include "param_sgd.h"
#include "param_mom.h"
#include <functional>
#include <vector>

using aurora::data::ptr;
using aurora::data::ref;
using std::function;
using std::vector;
using aurora::optimization::param;
using aurora::optimization::param_sgd;
using aurora::optimization::param_mom;
using aurora::math::tensor;

namespace aurora {
	namespace modeling {
		class model {
		public:
			ptr<tensor> x_ptr = new tensor(0);
			ptr<tensor> y_ptr = new tensor(0);
			ptr<tensor> x_grad_ptr = new tensor(0);
			ptr<tensor> y_grad_ptr = new tensor(0);

		public:
			tensor& x();
			tensor& y();
			tensor& x_grad();
			tensor& y_grad();

		public:
			virtual ~model();
			model();
			model(vector<param*>& pl);
			model(vector<param_sgd*>& pl);
			model(vector<param_mom*>& pl);

		public:
			virtual model* clone();
			virtual model* clone(vector<param*>& pl);
			virtual model* clone(vector<param_sgd*>& pl);
			virtual model* clone(vector<param_mom*>& pl);

		public:
			virtual void fwd();
			virtual void bwd();
			virtual void recur(function<void(model*)> func);

		public:
			virtual void prepend(model& other);
			virtual void append(model& other);
			virtual void compile();

		};
	}
}