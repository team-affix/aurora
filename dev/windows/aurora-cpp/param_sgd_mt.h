#pragma once
#include "param_sgd.h"
#include <mutex>

using aurora::params::param_sgd;
using std::mutex;

namespace aurora {
	namespace params {
		class param_sgd_mt : public param_sgd {
		public:
			ptr<double> gradient_ptr = new double(0);

		public:
			mutex accum_grad_mutex;

		public:
			virtual ~param_sgd_mt();
			param_sgd_mt();
			param_sgd_mt(double a_state, double a_drop_chance, double a_learn_rate, double a_gradient);

		public:
			virtual void accum_grad(double a_grad);

		public:
			virtual param* clone();

		};
	}
}