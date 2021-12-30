#include "affix-base/pch.h"
#include "pseudo_tnn.h"
#include "static_vals.h"
#include "layer.h"
#include "weight_junction.h"
#include "pseudo_tnn_hyperparams.h"
#include "neuron.h"
#include "param_init.h"

using namespace aurora;
using aurora::pseudo::tnn_hyperparams;
using models::layer;
using models::weight_junction;
using namespace aurora::models;
using std::vector;
using namespace aurora::params;
using std::uniform_real_distribution;

Sequential pseudo::tnn(vector<size_t> a_dims, Model a_neuron_template) {
	Sequential result = new sequential();
	for (int i = 0; i < a_dims.size() - 1; i++) {
		result->m_models.push_back(new layer(a_dims[i], a_neuron_template));
		result->m_models.push_back(new weight_junction(a_dims[i], a_dims[i + 1]));
	}
	result->m_models.push_back(new layer(a_dims.back(), a_neuron_template));
	return result;
}

Sequential pseudo::tnn(vector<size_t> a_dims, vector<Model> a_neuron_templates) {
	Sequential result = new sequential();
	for (int i = 0; i < a_dims.size() - 1; i++) {
		result->m_models.push_back(new layer(a_dims[i], a_neuron_templates[i]));
		result->m_models.push_back(new weight_junction(a_dims[i], a_dims[i + 1]));
	}
	result->m_models.push_back(new layer(a_dims.back(), a_neuron_templates.back()));
	return result;
}

Sequential pseudo::tnn_no_output(vector<size_t> a_dims, Model a_neuron_template) {
	Sequential result = new sequential();
	for (int i = 0; i < a_dims.size() - 1; i++) {
		result->m_models.push_back(new layer(a_dims[i], a_neuron_template));
		result->m_models.push_back(new weight_junction(a_dims[i], a_dims[i + 1]));
	}
	return result;
}

Sequential pseudo::tnn_no_output(vector<size_t> a_dims, vector<Model> a_neuron_templates) {
	Sequential result = new sequential();
	for (int i = 0; i < a_dims.size() - 1; i++) {
		result->m_models.push_back(new layer(a_dims[i], a_neuron_templates[i]));
		result->m_models.push_back(new weight_junction(a_dims[i], a_dims[i + 1]));
	}
	return result;
}

Sequential pseudo::tnn_compiled(vector<size_t> a_dims, param_vector& a_param_vec) {

	Sequential result = pseudo::tnn(a_dims, pseudo::nlr(0.3));

	size_t param_count = 0;
	result->param_recur(pseudo::param_count(param_count));

	double learn_rate = 0;
	double beta = 0;
	uniform_real_distribution<double> urd;

	tnn_hyperparams(param_count, learn_rate, beta, urd);

	result->param_recur(pseudo::param_init(new param_mom(learn_rate, beta), a_param_vec));

	result->compile();

	a_param_vec.randomize();
	a_param_vec.normalize();

	return result;
	
}
