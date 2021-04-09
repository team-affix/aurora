#pragma once
#include "ptr.h"
#include <vector>
#include <functional>
#include <random>

using aurora::data::ptr;
using std::vector;
using std::function;
using std::uniform_real_distribution;

namespace aurora {
	namespace params {
		class param_sgd;
		class param_mom;
		class param {
		protected:
			static uniform_real_distribution<double> drop_urd;
			ptr<double> dropped_state_ptr = new double(0);
			ptr<double> drop_chance_ptr = new double(0);
			bool dropped = false;

		public:
			ptr<double> state_ptr = new double(0);

		public:
			virtual ~param();
			param();
			param(double a_state, double a_drop_chance);

		protected:
			virtual bool should_drop();

		public:
			virtual double& state();
			virtual double& dropped_state();
			virtual double& drop_chance();

		public:
			virtual void drop();
			virtual void update();
			virtual param* clone();

		};
	}
}