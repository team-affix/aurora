#include "pch.h"
#include "tnn.h"
#include "layer.h"
#include "weight_junction.h"

using namespace aurora;
using models::layer;
using models::weight_junction;

sequential* pseudo::tnn(vector<size_t> a_npl, ptr<model> a_neuron_template, function<void(ptr<param>&)> a_init) {
	sequential* result = new sequential();
	for (int i = 0; i < a_npl.size() - 1; i++) {
		result->models.push_back(new layer(a_npl[i], a_neuron_template, a_init));
		result->models.push_back(new weight_junction(a_npl[i], a_npl[i + 1], a_init));
	}
	result->models.push_back(new layer(a_npl.back(), a_neuron_template, a_init));
	return result;
}

sequential* pseudo::tnn(vector<size_t> a_npl, vector<ptr<model>> a_neuron_templates, function<void(ptr<param>&)> a_init) {
	sequential* result = new sequential();
	for (int i = 0; i < a_npl.size() - 1; i++) {
		result->models.push_back(new layer(a_npl[i], a_neuron_templates[i], a_init));
		result->models.push_back(new weight_junction(a_npl[i], a_npl[i + 1], a_init));
	}
	result->models.push_back(new layer(a_npl.back(), a_neuron_templates.back(), a_init));
	return result;
}