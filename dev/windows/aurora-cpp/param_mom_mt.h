#pragma once
#include "param_mom.h"
#include <mutex>

using aurora::optimization::param_mom;
using std::mutex;
using std::lock_guard;

namespace aurora {
	namespace optimization {
		class param_mom_mt : public param_mom {
		public:
			ptr<double> momentum_ptr = new double(0);
			ptr<double> beta_ptr = new double(0);

		public:
			mutex state_mtx;
			mutex learn_rate_mtx;
			mutex gradient_mtx;
			mutex momentum_mtx;
			mutex beta_mtx;

		public:
			virtual ~param_mom_mt();
			param_mom_mt();
			param_mom_mt(double a_state, double a_learn_rate, double a_gradient, double a_momentum, double a_beta);

		public:
			virtual double& state();
			virtual double& learn_rate();
			virtual double& gradient();
			virtual double& momentum();
			virtual double& beta();

		public:
			virtual param* clone();

		};
	}
}