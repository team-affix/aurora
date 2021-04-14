#pragma once
#include "ptr.h"
#include <vector>
#include <functional>

using aurora::data::ptr;
using std::vector;
using std::function;

namespace aurora {
	namespace params {
		class param_sgd;
		class param_mom;
		class param {
		public:
			ptr<double> state_ptr = new double(0);

		public:
			virtual ~param();
			param();
			param(double a_state);

		public:
			virtual double& state();

		public:
			virtual void update();
			virtual param* clone();

		};
	}
}