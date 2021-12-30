#include "affix-base/pch.h"
#include "param_init.h"
#include "pseudo_lstm.h"
#include "pseudo_lstm_hyperparams.h"
#include "static_vals.h"
#include "pseudo_tnn.h"
#include "neuron.h"

using namespace aurora;
using namespace aurora::models;
using namespace aurora::params;
using std::uniform_real_distribution;

Lstm pseudo::lstm_compiled(size_t a_units, size_t a_max_timesteps, param_vector& a_param_vec) {

	Lstm result = new lstm(a_units);

	result->prep(a_max_timesteps);
	result->compile();

	return result;

}

Stacked_recurrent pseudo::lstm_mdim(size_t a_x_units, size_t a_h_units, size_t a_y_units) {

	Stacked_recurrent result = new stacked_recurrent({
		new sync(pseudo::tnn({a_x_units, a_h_units}, pseudo::nlr(0.3))),
		new lstm(a_h_units),
		new sync(pseudo::tnn({a_h_units, a_y_units}, pseudo::nlr(0.3))),
	});

	return result;

}

Stacked_recurrent pseudo::lstm_mdim_compiled(size_t a_x_units, size_t a_h_units, size_t a_y_units, size_t a_max_timesteps, param_vector& a_param_vec) {

	Stacked_recurrent result = lstm_mdim(a_x_units, a_h_units, a_y_units);

	size_t param_count = 0;
	result->param_recur(pseudo::param_count(param_count));

	double learn_rate = 0;
	double beta = 0;
	uniform_real_distribution<double> urd;

	lstm_hyperparams(param_count, learn_rate, beta, urd);

	result->param_recur(pseudo::param_init(new param_mom(learn_rate, beta), a_param_vec));

	result->prep(a_max_timesteps);
	result->compile();

	a_param_vec.randomize();
	a_param_vec.normalize();

	return result;

}

Stacked_recurrent pseudo::lstm_stacked(size_t a_units, size_t a_stack_height) {
	
	Stacked_recurrent result = new stacked_recurrent(a_stack_height, new lstm(a_units));

	return result;


}

Stacked_recurrent pseudo::lstm_stacked_compiled(size_t a_units, size_t a_stack_height, size_t a_max_timesteps, param_vector& a_param_vec) {

	Stacked_recurrent result = lstm_stacked(a_units, a_stack_height);

	size_t param_count = 0;
	result->param_recur(pseudo::param_count(param_count));

	double learn_rate = 0;
	double beta = 0;
	uniform_real_distribution<double> urd;

	lstm_hyperparams(param_count, learn_rate, beta, urd);

	result->param_recur(pseudo::param_init(new param_mom(learn_rate, beta), a_param_vec));

	result->prep(a_max_timesteps);
	result->compile();

	a_param_vec.randomize();
	a_param_vec.normalize();

	return result;

}

Stacked_recurrent pseudo::lstm_stacked_mdim(size_t a_x_units, size_t a_h_units, size_t a_y_units, size_t a_stack_height) {

	Stacked_recurrent result = new stacked_recurrent({
		new sync(pseudo::tnn({a_x_units, a_h_units}, pseudo::nlr(0.3))),
		lstm_stacked(a_h_units, a_stack_height),
		new sync(pseudo::tnn({a_h_units, a_y_units}, pseudo::nlr(0.3))),
	});

	return result;

}

Stacked_recurrent pseudo::lstm_stacked_mdim_compiled(size_t a_x_units, size_t a_h_units, size_t a_y_units, size_t a_stack_height, size_t a_max_timesteps, param_vector& a_param_vec) {

	Stacked_recurrent result = new stacked_recurrent({
		new sync(pseudo::tnn({a_x_units, a_h_units}, pseudo::nlr(0.3))),
		lstm_stacked(a_h_units, a_stack_height),
		new sync(pseudo::tnn({a_h_units, a_y_units}, pseudo::nlr(0.3))),
	});

	size_t param_count = 0;
	result->param_recur(pseudo::param_count(param_count));

	double learn_rate = 0;
	double beta = 0;
	uniform_real_distribution<double> urd;

	lstm_hyperparams(param_count, learn_rate, beta, urd);

	result->param_recur(pseudo::param_init(new param_mom(learn_rate, beta), a_param_vec));

	result->prep(a_max_timesteps);
	result->compile();

	a_param_vec.randomize();
	a_param_vec.normalize();

	return result;

}