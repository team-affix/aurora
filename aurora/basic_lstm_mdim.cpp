#include "pch.h"
#include "basic_lstm_mdim.h"
#include "basic_lstm_hyperparams.h"
#include "static_vals.h"

using namespace aurora;

Stacked_recurrent basic::basic_lstm_mdim(size_t a_x_units, size_t a_h_units, size_t a_y_units, size_t a_max_timesteps, param_vector& a_param_vec) {

	Stacked_recurrent result = new stacked_recurrent({
		new sync(pseudo::tnn({a_x_units, a_h_units}, pseudo::nlr(0.3))),
		new lstm(a_h_units),
		new sync(pseudo::tnn({a_h_units, a_y_units}, pseudo::nlr(0.3))),
	});

	size_t param_count = 0;
	result->param_recur(PARAM_COUNT(param_count));

	double learn_rate = 0;
	double beta = 0;
	uniform_real_distribution<double> urd;

	basic_lstm_hyperparams(param_count, learn_rate, beta, urd);

	result->param_recur(PARAM_INIT(param_mom(urd(aurora::static_vals::random_engine), learn_rate, 0, 0, beta), a_param_vec));

	result->prep(a_max_timesteps);
	result->compile();

	return result;

}