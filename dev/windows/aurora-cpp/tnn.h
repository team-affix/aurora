#pragma once
#include "model.h"
#include "sequential.h"
#include <vector>

using aurora::modeling::model;
using aurora::modeling::sequential;
using std::vector;

namespace aurora {
	namespace pseudo {
		sequential* tnn(vector<size_t> a_npl, ptr<model> a_neuron_template, vector<param*>& a_pl);
		sequential* tnn(vector<size_t> a_npl, ptr<model> a_neuron_template, vector<param_sgd*>& a_pl);
		sequential* tnn(vector<size_t> a_npl, ptr<model> a_neuron_template, vector<param_mom*>& a_pl);
		sequential* tnn(vector<size_t> a_npl, vector<ptr<model>> a_neuron_templates, vector<param*>& a_pl);
		sequential* tnn(vector<size_t> a_npl, vector<ptr<model>> a_neuron_templates, vector<param_sgd*>& a_pl);
		sequential* tnn(vector<size_t> a_npl, vector<ptr<model>> a_neuron_templates, vector<param_mom*>& a_pl);
	}
}