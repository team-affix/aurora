#include "pch.h"
#include "spc.h"
#include "static_vals.h"

using aurora::models::model;
using aurora::models::spc;

uniform_real_distribution<double> spc::m_urd(0, 1);

spc::~spc() {

}

spc::spc() {

}

spc::spc(size_t a_units) {
	units = a_units;
}

void spc::param_recur(function<void(Param&)> a_func) {

}

model* spc::clone(function<Param(Param&)> a_func) {
	spc* result = new spc(units);
	return result;
}

void spc::fwd() {
	y.clear();
	int l_selected_index = collapse(x);
	y[l_selected_index].val() = 1;
}

void spc::bwd() {

}

void spc::signal(const tensor& a_y_des) {
	y.sub_1d(a_y_des, y_grad);
}

void spc::model_recur(function<void(model*)> a_func) {
	a_func(this);
}

void spc::compile() {
	x = tensor::new_1d(units);
	x_grad = tensor::new_1d(units);
	y = tensor::new_1d(units);
	y_grad = tensor::new_1d(units);

	x_grad.group_join(y_grad);
}

int spc::collapse(const tensor& a_probability_tensor) {
	tensor result = tensor::new_1d(a_probability_tensor.size());
	double random_value = m_urd(static_vals::random_engine);
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
