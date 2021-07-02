#pragma once
#include "pch.h"

using std::uniform_real_distribution;

namespace aurora {
	namespace pseudo {
		void lstm_hyperparams(size_t a_param_count, double& a_learn_rate, double& a_beta, uniform_real_distribution<double>& a_param_urd);
	}
}
