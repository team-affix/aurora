#include "pch.h"
#include "basic_tnn.h"
#include "static_vals.h"
#include "basic_hyperparams.h"

using namespace aurora;
using aurora::basic::basic_hyperparams;

Sequential basic::tnn(vector<size_t> a_dims, param_vector& a_param_vec) {

	Sequential result = pseudo::tnn(a_dims, pseudo::nlr(0.3));

	size_t param_count = 0;
	result->param_recur(PARAM_COUNT(param_count));

	double learn_rate = 0;
	double beta = 0;
	uniform_real_distribution<double> urd;

	basic_hyperparams(param_count, learn_rate, beta, urd);

	result->param_recur(PARAM_INIT(param_mom(urd(aurora::static_vals::random_engine), learn_rate, 0, 0, beta), a_param_vec));

	result->compile();

	return result;
	
}
