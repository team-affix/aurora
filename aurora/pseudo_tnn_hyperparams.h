#pragma once
#include "affix-base/pch.h"

namespace aurora {
	namespace pseudo {
		void tnn_hyperparams(
			size_t a_param_count,
			double& a_learn_rate,
			double& a_beta,
			std::uniform_real_distribution<double>& a_param_urd);
	}
}
