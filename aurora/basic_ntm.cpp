#include "pch.h"
#include "basic_ntm.h"
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

Ntm basic::ntm_compiled(
	size_t a_memory_height,
	size_t a_memory_width,
	size_t a_num_readers,
	size_t a_num_writers,
	vector<int> a_valid_shifts,
	vector<size_t> a_head_hidden_dims,
	size_t a_max_timesteps,
	param_vector& a_param_vec) {

	double learn_rate = 0;
	double beta = 0;
	uniform_real_distribution<double> param_urd;

	Ntm result = new ntm(a_memory_height, a_memory_width, a_num_readers, a_num_writers, a_valid_shifts, a_head_hidden_dims);

	size_t param_count = 0;
	result->param_recur(PARAM_COUNT(param_count));

	ntm_hyperparams(param_count, a_memory_height, a_memory_width, a_num_readers, a_num_writers, learn_rate, beta, param_urd);

	result->param_recur(PARAM_INIT(param_mom(param_urd(static_vals::random_engine), learn_rate, 0, 0, beta), a_param_vec));

	result->prep(a_max_timesteps);
	result->compile();

	return result;
}

Stacked_recurrent basic::ntm_mdim(
	size_t a_x_units,
	size_t a_y_units,
	size_t a_memory_height,
	size_t a_memory_width,
	size_t a_num_readers,
	size_t a_num_writers,
	vector<int> a_valid_shifts,
	vector<size_t> a_head_hidden_dims,
	param_vector& a_param_vec) {

	double learn_rate = 0;
	double beta = 0;
	uniform_real_distribution<double> param_urd;

	Sync in = new sync(pseudo::tnn({ a_x_units, a_memory_width }, pseudo::nlr(0.3)));
	Ntm mid = new ntm(a_memory_height, a_memory_width, a_num_readers, a_num_writers, a_valid_shifts, a_head_hidden_dims);
	Sync out = new sync(pseudo::tnn({ a_memory_width, a_y_units }, pseudo::nlr(0.3)));

	Stacked_recurrent result = new stacked_recurrent({ in, mid, out });

	size_t param_count = 0;
	result->param_recur(PARAM_COUNT(param_count));

	ntm_hyperparams(param_count, a_memory_height, a_memory_width, a_num_readers, a_num_writers, learn_rate, beta, param_urd);

	result->param_recur(PARAM_INIT(param_mom(param_urd(static_vals::random_engine), learn_rate, 0, 0, beta), a_param_vec));

	return result;

}

Stacked_recurrent basic::ntm_mdim_compiled(
	size_t a_x_units,
	size_t a_y_units,
	size_t a_memory_height,
	size_t a_memory_width,
	size_t a_num_readers,
	size_t a_num_writers,
	vector<int> a_valid_shifts,
	vector<size_t> a_head_hidden_dims,
	size_t a_max_timesteps,
	param_vector& a_param_vec) {

	Stacked_recurrent result = ntm_mdim(a_x_units, a_y_units, a_memory_height, a_memory_width, a_num_readers, a_num_writers, a_valid_shifts, a_head_hidden_dims, a_param_vec);

	result->prep(a_max_timesteps);
	result->compile();

	return result;

}
