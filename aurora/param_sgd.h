#pragma once
#include "affix-base/pch.h"
#include "param.h"

namespace aurora {
	namespace params {
		class param_sgd : public aurora::params::param {
		public:
			affix_base::data::ptr<double> m_learn_rate_ptr = new double(0);
			affix_base::data::ptr<double> m_gradient_ptr = new double(0);

		public:
			virtual ~param_sgd();
			param_sgd();
			param_sgd(
				double a_learn_rate
			);

		public:
			virtual double& learn_rate();
			virtual const double& learn_rate() const;

		public:
			virtual double& gradient();
			virtual const double& gradient() const;


		public:
			virtual void accum_grad(
				double a_grad
			);

		public:
			virtual void update();

		public:
			virtual param* clone() const;

		};
		typedef affix_base::data::ptr<param_sgd> Param_sgd;
	}
}
