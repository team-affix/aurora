#pragma once
#include "param.h"

using aurora::optimization::param;

namespace aurora {
	namespace optimization {
		class param_sgd : public param {
		public:
			ptr<double> gradient_ptr = new double(0);

		public:
			virtual ~param_sgd();
			param_sgd();
			param_sgd(double a_state, double a_learn_rate, double a_gradient);

		public:
			virtual double& gradient();

		public:
			virtual param* clone();

		};
	}
}