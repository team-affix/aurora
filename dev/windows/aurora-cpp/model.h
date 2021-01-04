#pragma once
#include "ptr.h"
#include "complex.h"
#include "param.h"
#include "param_sgd.h"
#include "param_mom.h"
#include <functional>
#include <vector>

using std::function;
using std::vector;
using aurora::optimization::param;
using aurora::optimization::param_sgd;
using aurora::optimization::param_mom;
using aurora::math::complex;

namespace aurora {
	namespace modeling {
		class model {
		public:
			complex x;
			complex x_grad;
			complex y;
			complex y_grad;

		public:
			virtual ~model();
			model();
			model(vector<param*>& pl);
			model(vector<param_sgd*>& pl);
			model(vector<param_mom*>& pl);

		public:
			virtual void fwd();
			virtual void bwd();
			virtual void recur(function<void(model&)> func);
			virtual void prepend(model& other);
			virtual void append(model& other);
			virtual void compile();

		};
	}
}