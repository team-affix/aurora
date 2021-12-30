#pragma once
#include "affix-base/pch.h"
#include "param_sgd.h"

namespace aurora {
	namespace params {
		class param_mom : public param_sgd {
		public:
			affix_base::data::ptr<double> m_momentum_ptr = new double(0);
			affix_base::data::ptr<double> m_beta_ptr = new double(0);

		public:
			virtual ~param_mom();
			param_mom();
			param_mom(
				double a_learn_rate,
				double a_beta
			);

		public:
			virtual double& momentum();
			virtual const double& momentum() const;

		public:
			virtual double& beta();
			virtual const double& beta() const;

		public:
			virtual void update();

		public:
			virtual param* clone() const;

		};
		typedef affix_base::data::ptr<param_mom> Param_mom;
	}
}
