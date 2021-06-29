#pragma once
#include "pch.h"

using std::uniform_real_distribution;

namespace aurora {
	namespace basic {
		void ntm_hyperparams(size_t a_param_count, size_t a_memory_height, size_t a_memory_width, size_t a_num_readers, size_t a_num_writers, double& a_learn_rate, double& a_beta, uniform_real_distribution<double>& a_param_urd);
	}
}
