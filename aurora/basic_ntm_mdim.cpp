#include "pch.h"
#include "basic_ntm_mdim.h"
#include "ntm.h"
#include "sync.h"
#include "basic_ntm_hyperparams.h"
#include "static_vals.h"

using aurora::models::stacked_recurrent;
using aurora::models::ntm;
using aurora::models::Ntm;
using aurora::models::sync;
using aurora::models::Sync;
using namespace aurora;

Stacked_recurrent basic::ntm_mdim(size_t a_x_units, size_t a_h_units, size_t a_y_units, size_t a_memory_height, size_t a_complexity, size_t a_max_timesteps, param_vector& a_param_vec) {

	double learn_rate = 0;
	double beta = 0;
	uniform_real_distribution<double> param_urd;

	size_t num_readers = a_complexity;
	size_t num_writers = a_complexity;

	Sync in = new sync(pseudo::tnn({ a_x_units, a_h_units }, pseudo::nlr(0.3)));
	Ntm mid = new ntm(a_memory_height, a_h_units, num_readers, num_writers, { -1, 0, 1 }, { a_h_units, 2 * a_h_units });
	Sync out = new sync(pseudo::tnn({ a_h_units, a_y_units }, pseudo::nlr(0.3)));

	Stacked_recurrent result = new stacked_recurrent({ in, mid, out });

	size_t param_count = 0;
	result->param_recur(PARAM_COUNT(param_count));

	basic_ntm_hyperparams(param_count, learn_rate, beta, param_urd);

	result->param_recur(PARAM_INIT(param_mom(param_urd(static_vals::random_engine), learn_rate, 0, 0, beta), a_param_vec));

	result->prep(a_max_timesteps);
	result->compile();

	return result;

}
