#pragma once
#include "param.h"
#include "param_sgd.h"
#include "param_mom.h"
#include <vector>

using aurora::optimization::param;
using aurora::optimization::param_sgd;
using aurora::optimization::param_mom;
using std::vector;

namespace aurora {
	namespace pseudo {
		void init(vector<param*>& a_pl, int a_rnd_init, double a_state_min, double a_state_max, double a_learn_rate);
		void init(vector<param_sgd*>& a_pl, int a_rnd_init, double a_state_min, double a_state_max, double a_learn_rate);
		void init(vector<param_mom*>& a_pl, int a_rnd_init, double a_state_min, double a_state_max, double a_learn_rate, double a_beta);
	}
}