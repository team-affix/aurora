#pragma once
#include "pch.h"
#include "model.h"
#include "sequential.h"

using aurora::models::model;
using aurora::models::sequential;
using std::vector;

namespace aurora {
	namespace pseudo {
		sequential* tnn(vector<size_t> a_npl, ptr<model> a_neuron_template, function<void(ptr<param>&)> a_init);
		sequential* tnn(vector<size_t> a_npl, vector<ptr<model>> a_neuron_templates, function<void(ptr<param>&)> a_init);
	}
}