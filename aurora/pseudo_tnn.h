#pragma once
#include "pch.h"
#include "sequential.h"
#include "param_vector.h"
#include "model.h"

using aurora::models::Sequential;
using aurora::models::sequential;
using aurora::models::Model;
using aurora::params::param_vector;
using std::vector;

namespace aurora {
	namespace pseudo {
		Sequential tnn(vector<size_t> a_dims, Model a_neuron_template);
		Sequential tnn(vector<size_t> a_dims, vector<Model> a_neuron_templates);
		Sequential tnn_no_output(vector<size_t> a_dims, Model a_neuron_template);
		Sequential tnn_no_output(vector<size_t> a_dims, vector<Model> a_neuron_templates);
		Sequential tnn_compiled(vector<size_t> a_dims, param_vector& a_param_vec);
	}
}
