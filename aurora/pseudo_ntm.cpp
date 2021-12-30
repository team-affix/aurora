#include "affix-base/pch.h"
#include "pseudo_ntm.h"
#include "ntm.h"
#include "sync.h"
#include "pseudo_ntm_hyperparams.h"
#include "static_vals.h"
#include "param_init.h"
#include "lstm.h"
#include "pseudo_tnn.h"
#include "neuron.h"

using aurora::models::stacked_recurrent;
using aurora::models::ntm;
using aurora::models::Ntm;
using aurora::models::sync;
using aurora::models::Sync;
using namespace aurora;
using std::vector;
using aurora::maths::tensor;
using aurora::params::param_vector;
using std::uniform_real_distribution;
using namespace aurora::params;
using namespace aurora::models;

Ntm pseudo::ntm_compiled(
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
	result->param_recur(pseudo::param_count(param_count));

	pseudo::ntm_hyperparams(param_count, learn_rate, beta, param_urd);

	result->param_recur(pseudo::param_init(new param_mom(learn_rate, beta), a_param_vec));

	result->prep(a_max_timesteps);
	result->compile();

	a_param_vec.randomize();
	a_param_vec.normalize();

	return result;
}

Stacked_recurrent pseudo::ntm_mdim(
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
	result->param_recur(pseudo::param_count(param_count));

	pseudo::ntm_hyperparams(param_count, learn_rate, beta, param_urd);

	result->param_recur(pseudo::param_init(new param_mom(learn_rate, beta), a_param_vec));

	a_param_vec.randomize();
	a_param_vec.normalize();

	return result;

}

Stacked_recurrent pseudo::ntm_mdim_compiled(
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

	Stacked_recurrent result = pseudo::ntm_mdim(a_x_units, a_y_units, a_memory_height, a_memory_width, a_num_readers, a_num_writers, a_valid_shifts, a_head_hidden_dims, a_param_vec);

	result->prep(a_max_timesteps);
	result->compile();

	return result;

}
