#pragma once
#include "ptr.h"
#include <vector>

using aurora::data::ptr;
using std::vector;

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

		public:
			double& state();
			double& learn_rate();

		public:
			virtual param* clone();
			virtual param* to_param();
			virtual param_sgd* to_param_sgd();
			virtual param_mom* to_param_mom();

		};
	}
}