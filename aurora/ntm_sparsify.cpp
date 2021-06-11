#include "pch.h"
#include "ntm_sparsify.h"

using aurora::models::ntm_sparsify;

ntm_sparsify::~ntm_sparsify() {

}

ntm_sparsify::ntm_sparsify() {

}

ntm_sparsify::ntm_sparsify(size_t a_memory_height) {
	memory_height = a_memory_height;
}

void ntm_sparsify::pmt_wise(function<void(ptr<param>&)> a_func) {

}

model* ntm_sparsify::clone() {
	ntm_sparsify* result = new ntm_sparsify();
	result->memory_height = memory_height;
	return result;
}

model* ntm_sparsify::clone(function<void(ptr<param>&)> a_func) {
	ntm_sparsify* result = new ntm_sparsify();
	result->memory_height = memory_height;
	return result;
}

void ntm_sparsify::fwd() {
	for (int i = 0; i < memory_height; i++)
		y[i].val() = exp(beta[0] * x[i].val());
}

void ntm_sparsify::bwd() {
	beta_grad[0].val() = 0;
	for (int i = 0; i < memory_height; i++) {
		x_grad[i].val() = y_grad[i] * y[i] * beta[0];
		beta_grad[0].val() += y_grad[i] * y[i] * x[i];
	}
}

void ntm_sparsify::signal(tensor& a_y_des) {
	y.sub_1d(a_y_des, y_grad);
}

void ntm_sparsify::recur(function<void(model*)> a_func) {
	a_func(this);
}

void ntm_sparsify::compile() {
	x = tensor::new_1d(memory_height);
	x_grad = tensor::new_1d(memory_height);
	y = tensor::new_1d(memory_height);
	y_grad = tensor::new_1d(memory_height);
	beta = tensor::new_1d(1);
	beta_grad = tensor::new_1d(1);
}