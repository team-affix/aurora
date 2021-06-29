#include "basic_ntm_hyperparams.h"

using namespace aurora;

void basic::ntm_hyperparams(size_t a_param_count, size_t a_memory_height, size_t a_memory_width, size_t a_num_readers, size_t a_num_writers, double& a_learn_rate, double& a_beta, uniform_real_distribution<double>& a_param_urd) {

	a_learn_rate = 10.0 / (double)a_param_count / (double)a_num_readers / (double)a_num_writers / (double)a_memory_height / (double)a_memory_width;
	a_beta = 0.9;
	a_param_urd = uniform_real_distribution<double>(-1.0, 1.0);
}
