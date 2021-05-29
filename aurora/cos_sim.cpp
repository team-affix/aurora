#include "pch.h"
#include "cos_sim.h"

using aurora::models::cos_sim;

cos_sim::~cos_sim() {

}

cos_sim::cos_sim() {

}

cos_sim::cos_sim(size_t a_units) {
	units = a_units;
}

void cos_sim::pmt_wise(function<void(ptr<param>&)> a_func) {

}

model* cos_sim::clone() {
	cos_sim* result = new cos_sim();
	result->units = units;
	result->magnitude_0 = magnitude_0;
	result->magnitude_1 = magnitude_1;
	result->magnitude_product = magnitude_product;
	result->dot_product = dot_product;
	return result;
}

model* cos_sim::clone(function<void(ptr<param>&)> a_func) {
	cos_sim* result = new cos_sim();
	result->units = units;
	result->magnitude_0 = magnitude_0;
	result->magnitude_1 = magnitude_1;
	result->magnitude_product = magnitude_product;
	result->dot_product = dot_product;
	return result;
}

void cos_sim::fwd() {
	magnitude_0 = x[0].mag_1d();
	magnitude_1 = x[1].mag_1d();
	magnitude_product = magnitude_0 * magnitude_1;
	dot_product = x[0].dot_1d(x[1]);
	y.val() = dot_product / magnitude_product;
}

void cos_sim::bwd() {
	for (int i = 0; i < x.width(); i++) {
		double x_0_a = x[1][i];
		double x_0_b = x[0][i] * dot_product;
		double x_0_c = magnitude_product;
		double x_0_d = magnitude_1 * pow(magnitude_0, 3);
		x_grad[0][i].val() = y_grad * (x_0_a / x_0_c - x_0_b / x_0_d);
		double x_1_a = x[0][i];
		double x_1_b = x[1][i] * dot_product;
		double x_1_c = magnitude_product;
		double x_1_d = magnitude_0 * pow(magnitude_1, 3);
		x_grad[1][i].val() = y_grad * (x_1_a / x_1_c - x_1_b / x_1_d);
	}
}

tensor& cos_sim::fwd(tensor& a_x) {
	x.pop(a_x);
	fwd();
	return y;
}

tensor& cos_sim::bwd(tensor& a_y_grad) {
	y_grad.pop(a_y_grad);
	bwd();
	return x_grad;
}

void cos_sim::signal(tensor& a_y_des) {
	y_grad.val() = y.val() - a_y_des.val();
}

void cos_sim::cycle(tensor& a_x, tensor& a_y_des) {
	x.pop(a_x);
	fwd();
	signal(a_y_des);
	bwd();
}

void cos_sim::recur(function<void(model*)> a_func) {
	a_func(this);
}

void cos_sim::compile() {
	x = tensor::new_2d(2, units);
	x_grad = tensor::new_2d(2, units);
}