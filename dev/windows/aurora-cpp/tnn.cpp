#include "tnn.h"
#include "layer.h"
#include "weight_junction.h"

using namespace aurora;
using modeling::layer;
using modeling::weight_junction;

sequential* pseudo::tnn(vector<size_t> a_npl, ptr<model> a_neuron_template, vector<param*>& a_pl) {
	sequential* result = new sequential();
	for (int i = 0; i < a_npl.size() - 1; i++) {
		result->models.push_back(new layer(a_npl[i], a_neuron_template, a_pl));
		result->models.push_back(new weight_junction(a_npl[i], a_npl[i + 1], a_pl));
	}
	result->models.push_back(new layer(a_npl.back(), a_neuron_template, a_pl));
	return result;
}

sequential* pseudo::tnn(vector<size_t> a_npl, ptr<model> a_neuron_template, vector<param_sgd*>& a_pl) {
	sequential* result = new sequential();
	for (int i = 0; i < a_npl.size() - 1; i++) {
		result->models.push_back(new layer(a_npl[i], a_neuron_template, a_pl));
		result->models.push_back(new weight_junction(a_npl[i], a_npl[i + 1], a_pl));
	}
	result->models.push_back(new layer(a_npl.back(), a_neuron_template, a_pl));
	return result;
}

sequential* pseudo::tnn(vector<size_t> a_npl, ptr<model> a_neuron_template, vector<param_mom*>& a_pl) {
	sequential* result = new sequential();
	for (int i = 0; i < a_npl.size() - 1; i++) {
		result->models.push_back(new layer(a_npl[i], a_neuron_template, a_pl));
		result->models.push_back(new weight_junction(a_npl[i], a_npl[i + 1], a_pl));
	}
	result->models.push_back(new layer(a_npl.back(), a_neuron_template, a_pl));
	return result;
}

sequential* pseudo::tnn(vector<size_t> a_npl, vector<ptr<model>> a_neuron_templates, vector<param*>& a_pl) {
	sequential* result = new sequential();
	for (int i = 0; i < a_npl.size() - 1; i++) {
		result->models.push_back(new layer(a_npl[i], a_neuron_templates[i], a_pl));
		result->models.push_back(new weight_junction(a_npl[i], a_npl[i + 1], a_pl));
	}
	result->models.push_back(new layer(a_npl.back(), a_neuron_templates.back(), a_pl));
	return result;
}

sequential* pseudo::tnn(vector<size_t> a_npl, vector<ptr<model>> a_neuron_templates, vector<param_sgd*>& a_pl) {
	sequential* result = new sequential();
	for (int i = 0; i < a_npl.size() - 1; i++) {
		result->models.push_back(new layer(a_npl[i], a_neuron_templates[i], a_pl));
		result->models.push_back(new weight_junction(a_npl[i], a_npl[i + 1], a_pl));
	}
	result->models.push_back(new layer(a_npl.back(), a_neuron_templates.back(), a_pl));
	return result;
}

sequential* pseudo::tnn(vector<size_t> a_npl, vector<ptr<model>> a_neuron_templates, vector<param_mom*>& a_pl) {
	sequential* result = new sequential();
	for (int i = 0; i < a_npl.size() - 1; i++) {
		result->models.push_back(new layer(a_npl[i], a_neuron_templates[i], a_pl));
		result->models.push_back(new weight_junction(a_npl[i], a_npl[i + 1], a_pl));
	}
	result->models.push_back(new layer(a_npl.back(), a_neuron_templates.back(), a_pl));
	return result;
}