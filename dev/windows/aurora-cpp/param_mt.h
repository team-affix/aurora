#pragma once
#include "param.h"
#include <mutex>

using aurora::optimization::param;
using std::mutex;
using std::lock_guard;

namespace aurora {
	namespace optimization {
		class param_mt : public param {
		public:
			ptr<double> state_ptr = new double(0);
			ptr<double> learn_rate_ptr = new double(0);

		public:
			mutex state_mtx;
			mutex learn_rate_mtx;

		public:
			virtual ~param_mt();
			param_mt();
			param_mt(double a_state, double a_learn_rate);

		public:
			double& state();
			double& learn_rate();

		public:
			virtual param* clone();

		};
	}
}