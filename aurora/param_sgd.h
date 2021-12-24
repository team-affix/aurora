#pragma once
#include "affix-base/pch.h"
#include "param.h"

namespace aurora {
	namespace params {
		class param_sgd : public aurora::params::param {
		public:
			affix_base::data::ptr<double> learn_rate_ptr = new double(0);
			affix_base::data::ptr<double> gradient_ptr = new double(0);

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
		typedef affix_base::data::ptr<param_sgd> Param_sgd;
	}
}
