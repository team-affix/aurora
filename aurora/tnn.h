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
		sequential* tnn(vector<size_t> a_dims, Model a_neuron_template, function<void(Param&)> a_func);
		sequential* tnn(vector<size_t> a_dims, vector<Model> a_neuron_templates, function<void(Param&)> a_func);
		sequential* tnn_no_output(vector<size_t> a_dims, Model a_neuron_template, function<void(Param&)> a_func);
		sequential* tnn_no_output(vector<size_t> a_dims, vector<Model> a_neuron_templates, function<void(Param&)> a_func);
	}
}