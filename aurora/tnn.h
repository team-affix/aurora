#pragma once
#include "pch.h"
#include "model.h"
#include "sequential.h"

using aurora::params::Param;
using aurora::models::model;
using aurora::models::Model;
using aurora::models::sequential;
using std::vector;

namespace aurora {
	namespace pseudo {
		sequential* tnn(vector<size_t> a_dims, ptr<model> a_neuron_template);
		sequential* tnn(vector<size_t> a_dims, vector<ptr<model>> a_neuron_templates);
		sequential* tnn_no_output(vector<size_t> a_dims, ptr<model> a_neuron_template);
		sequential* tnn_no_output(vector<size_t> a_dims, vector<ptr<model>> a_neuron_templates);
	}
}