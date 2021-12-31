#include "affix-base/pch.h"
#include "pseudo_tnn.h"
#include "static_vals.h"
#include "layer.h"
#include "weight_junction.h"
#include "neuron.h"
#include "param_init.h"

using namespace aurora;
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
