#pragma once
#include "param_mom.h"
#include <mutex>

using aurora::params::param_mom;
using std::mutex;
using std::lock_guard;

namespace aurora {
	namespace params {
		class param_mom_mt : public param_mom {
		public:
			ptr<double> momentum_ptr = new double(0);
			ptr<double> beta_ptr = new double(0);

		public:
			mutex accum_grad_mtx;

		public:
			virtual ~param_mom_mt();
			param_mom_mt();
			param_mom_mt(double a_state, double a_drop_chance, double a_learn_rate, double a_gradient, double a_momentum, double a_beta);

		public:
			virtual void accum_grad(double a_grad);

		public:
			virtual param* clone();

		};
	}
}