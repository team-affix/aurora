#include "pch.h"
#include "tnn.h"
#include "layer.h"
#include "weight_junction.h"

using namespace aurora;
using models::layer;
using models::weight_junction;

sequential* pseudo::tnn(vector<size_t> a_dims, ptr<model> a_neuron_template, function<void(ptr<param>&)> a_func) {
	sequential* result = new sequential();
	for (int i = 0; i < a_dims.size() - 1; i++) {
		result->models.push_back(new layer(a_dims[i], a_neuron_template, a_func));
		result->models.push_back(new weight_junction(a_dims[i], a_dims[i + 1], a_func));
	}
	result->models.push_back(new layer(a_dims.back(), a_neuron_template, a_func));
	return result;
}

sequential* pseudo::tnn(vector<size_t> a_dims, vector<ptr<model>> a_neuron_templates, function<void(ptr<param>&)> a_func) {
	sequential* result = new sequential();
	for (int i = 0; i < a_dims.size() - 1; i++) {
		result->models.push_back(new layer(a_dims[i], a_neuron_templates[i], a_func));
		result->models.push_back(new weight_junction(a_dims[i], a_dims[i + 1], a_func));
	}
	result->models.push_back(new layer(a_dims.back(), a_neuron_templates.back(), a_func));
	return result;
}