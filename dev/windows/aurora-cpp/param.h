#pragma once
#include "ptr.h"
#include <vector>
#include <functional>

using aurora::data::ptr;
using std::vector;
using std::function;

namespace aurora {
	namespace optimization {
		class param_sgd;
		class param_mom;
		class param {
		public:
			ptr<double> state_ptr = new double(0);
			ptr<double> learn_rate_ptr = new double(0);

		public:
			virtual ~param();
			param();
			param(double a_state, double a_learn_rate);

		public:
			virtual double& state();
			virtual double& learn_rate();

		public:
			virtual param* clone();

		};
	}
}