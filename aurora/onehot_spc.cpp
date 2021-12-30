#include "affix-base/pch.h"
#include "onehot_spc.h"
#include "static_vals.h"

using aurora::models::model;
using aurora::models::onehot_spc;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::params::param_sgd;
using aurora::maths::tensor;
using std::vector;
using std::initializer_list;
using std::uniform_real_distribution;

uniform_real_distribution<double> onehot_spc::s_urd(0, 1);

onehot_spc::~onehot_spc() {

}

onehot_spc::onehot_spc() {

}

onehot_spc::onehot_spc(size_t a_units) {
	m_units = a_units;
}

void onehot_spc::param_recur(const function<void(Param&)>& a_func) {

}

model* onehot_spc::clone(const function<Param(Param&)>& a_func) {
	onehot_spc* result = new onehot_spc(m_units);
	return result;
}

void onehot_spc::fwd() {
	m_y.clear();
	int l_selected_index = collapse(m_x);
	m_y[l_selected_index].val() = 1;
}

void onehot_spc::bwd() {

}

void onehot_spc::model_recur(const function<void(model*)>& a_func) {
	a_func(this);
}

void onehot_spc::compile() {
	m_x = tensor::new_1d(m_units);
	m_x_grad = tensor::new_1d(m_units);
	m_y = tensor::new_1d(m_units);
	m_y_grad = tensor::new_1d(m_units);

	m_x_grad.group_join(m_y_grad);
}

int onehot_spc::collapse(const tensor& a_probability_tensor) {
	tensor result = tensor::new_1d(a_probability_tensor.size());
	double random_value = s_urd(static_vals::random_engine);
	double bin_lower_bound = 0;
	double bin_upper_bound = 0;
	for (int i = 0; i < a_probability_tensor.size(); i++) {
		bin_upper_bound += a_probability_tensor[i];
		if (random_value >= bin_lower_bound && random_value <= bin_upper_bound) {
			return i;
		}
		bin_lower_bound += a_probability_tensor[i];
	}
	return -1;
}
