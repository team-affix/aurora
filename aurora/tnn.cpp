#include "pch.h"
#include "tnn.h"
#include "layer.h"
#include "weight_junction.h"

using namespace aurora;
using models::layer;
using models::weight_junction;

sequential* pseudo::tnn(vector<size_t> a_dims, Model a_neuron_template) {
	sequential* result = new sequential();
	for (int i = 0; i < a_dims.size() - 1; i++) {
		result->models.push_back(new layer(a_dims[i], a_neuron_template));
		result->models.push_back(new weight_junction(a_dims[i], a_dims[i + 1]));
	}
	result->models.push_back(new layer(a_dims.back(), a_neuron_template));
	return result;
}

sequential* pseudo::tnn(vector<size_t> a_dims, vector<Model> a_neuron_templates) {
	sequential* result = new sequential();
	for (int i = 0; i < a_dims.size() - 1; i++) {
		result->models.push_back(new layer(a_dims[i], a_neuron_templates[i]));
		result->models.push_back(new weight_junction(a_dims[i], a_dims[i + 1]));
	}
	result->models.push_back(new layer(a_dims.back(), a_neuron_templates.back()));
	return result;
}

sequential* pseudo::tnn_no_output(vector<size_t> a_dims, Model a_neuron_template) {
	sequential* result = new sequential();
	for (int i = 0; i < a_dims.size() - 1; i++) {
		result->models.push_back(new layer(a_dims[i], a_neuron_template));
		result->models.push_back(new weight_junction(a_dims[i], a_dims[i + 1]));
	}
	return result;
}

sequential* pseudo::tnn_no_output(vector<size_t> a_dims, vector<Model> a_neuron_templates) {
	sequential* result = new sequential();
	for (int i = 0; i < a_dims.size() - 1; i++) {
		result->models.push_back(new layer(a_dims[i], a_neuron_templates[i]));
		result->models.push_back(new weight_junction(a_dims[i], a_dims[i + 1]));
	}
	return result;
}