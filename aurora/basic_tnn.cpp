#include "pch.h"
#include "basic_tnn.h"
#include "static_vals.h"

using namespace aurora;

basic_model basic::tnn(vector<size_t> a_dims) {

	basic_model result;
	result.m_model = pseudo::tnn(a_dims, pseudo::nlr(0.3));

	size_t param_count = 0;
	result.m_model->param_recur(PARAM_COUNT(param_count));

	uniform_real_distribution<double> urd(-1.0 / (double)param_count, 1.0 / (double)param_count);

	result.m_model->param_recur(PARAM_INIT(param_mom(urd(aurora::static_vals::random_engine), 0.02, 0, 0, 0.9), result.m_params));

	result.m_model->compile();

	return result;
	
}
