#pragma once
#include "pch.h"
#include "param_sgd.h"

namespace aurora {
	namespace params {
		class param_mom : public param_sgd {
		public:
			ptr<double> momentum_ptr = new double(0);
			ptr<double> beta_ptr = new double(0);

		public:
			virtual ~param_mom();
			param_mom();
			param_mom(double a_state, double a_learn_rate, double a_gradient, double a_momentum, double a_beta);

		public:
			virtual double& momentum();
			virtual double& beta();

		public:
			virtual void update();
			virtual param* clone();

		};
	}
}