#include "pch.h"
#include "basic_tnn.h"
#include "static_vals.h"

using namespace aurora;

Model basic::tnn(vector<size_t> a_dims, param_vector& a_param_vec) {

	Model result = pseudo::tnn(a_dims, pseudo::nlr(0.3));

	size_t param_count = 0;
	result ->param_recur(PARAM_COUNT(param_count));

	uniform_real_distribution<double> urd(-1.0 / (double)param_count, 1.0 / (double)param_count);

	double learn_rate = 0.2 / (double)param_count;
	double beta = 0.9;

	result->param_recur(PARAM_INIT(param_mom(urd(aurora::static_vals::random_engine), learn_rate, 0, 0, beta), a_param_vec));

	result->compile();

	return result;
	
}
