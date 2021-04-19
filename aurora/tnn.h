#pragma once
#include "pch.h"
#include "model.h"
#include "sequential.h"

using aurora::models::model;
using aurora::models::sequential;
using std::vector;

namespace aurora {
	namespace pseudo {
		sequential* tnn(vector<size_t> a_dims, ptr<model> a_neuron_template, function<void(ptr<param>&)> a_func);
		sequential* tnn(vector<size_t> a_dims, vector<ptr<model>> a_neuron_templates, function<void(ptr<param>&)> a_func);
	}
}