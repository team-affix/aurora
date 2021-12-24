#include "affix-base/pch.h"
#include "pseudo_tnn_hyperparams.h"

using namespace aurora;
using std::uniform_real_distribution;

void pseudo::tnn_hyperparams(
	size_t a_param_count,
	double& a_learn_rate,
	double& a_beta,
	uniform_real_distribution<double>& a_param_urd) {

	a_learn_rate = 1.0 / (double)a_param_count;
	a_beta = 0.9;
	a_param_urd = uniform_real_distribution<double>(-1.0 / (double)a_param_count, 1.0 / (double)a_param_count);

}
