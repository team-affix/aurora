#pragma once
#include "pch.h"
#include "param.h"

using aurora::params::param;

namespace aurora {
	namespace params {
		class param_sgd : public param {
		public:
			ptr<double> learn_rate_ptr = new double(0);
			ptr<double> gradient_ptr = new double(0);

		public:
			virtual ~param_sgd();
			param_sgd();
			param_sgd(double a_state, double a_learn_rate, double a_gradient);

		public:
			virtual double& learn_rate();
			virtual double& gradient();

		public:
			virtual void accum_grad(double a_grad);

		public:
			virtual void update();
			virtual param* clone();

		};
	}
}