#pragma once
#include "affix-base/pch.h"
#include "sequential.h"
#include "param_vector.h"
#include "model.h"

namespace aurora {
	namespace pseudo {
		aurora::models::Sequential tnn(
			std::vector<size_t> a_dims,
			aurora::models::Model a_neuron_template
		);
		aurora::models::Sequential tnn(
			std::vector<size_t> a_dims,
			std::vector<aurora::models::Model> a_neuron_templates
		);
		aurora::models::Sequential tnn_no_output(
			std::vector<size_t> a_dims,
			aurora::models::Model a_neuron_template
		);
		aurora::models::Sequential tnn_no_output(
			std::vector<size_t> a_dims,
			std::vector<aurora::models::Model> a_neuron_templates
		);
		aurora::models::Sequential tnn_compiled(
			std::vector<size_t> a_dims,
			aurora::params::param_vector& a_param_vec
		);
	}
}
