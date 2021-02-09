#pragma once
#include "param_sgd.h"
#include <mutex>

using aurora::optimization::param_sgd;
using std::mutex;
using std::lock_guard;

namespace aurora {
	namespace optimization {
		class param_sgd_mt : public param_sgd {
		public:
			ptr<double> gradient_ptr = new double(0);

		public:
			mutex state_mtx;
			mutex learn_rate_mtx;
			mutex gradient_mtx;

		public:
			virtual ~param_sgd_mt();
			param_sgd_mt();
			param_sgd_mt(double a_state, double a_learn_rate, double a_gradient);

		public:
			virtual double& state();
			virtual double& learn_rate();
			virtual double& gradient();

		public:
			virtual param* clone();

		};
	}
}