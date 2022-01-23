#include "affix-base/pch.h"
#include "param_init.h"
#include "pseudo_lstm.h"
#include "static_vals.h"
#include "pseudo_tnn.h"
#include "neuron.h"

using namespace aurora;
using namespace aurora::models;
using namespace aurora::params;
using std::uniform_real_distribution;

Stacked_recurrent pseudo::lstm_mdim(size_t a_x_units, size_t a_h_units, size_t a_y_units) {

	return new stacked_recurrent({
		new sync(pseudo::tnn({a_x_units, a_h_units}, pseudo::nlr(0.3))),
		new lstm(a_h_units),
		new sync(pseudo::tnn({a_h_units, a_y_units}, pseudo::nlr(0.3))),
	});

}

Stacked_recurrent pseudo::lstm_stacked(size_t a_units, size_t a_stack_height) {
	
	Stacked_recurrent result = new stacked_recurrent(a_stack_height, new lstm(a_units));
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
