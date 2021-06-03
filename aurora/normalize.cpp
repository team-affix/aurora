#include "pch.h"
#include "normalize.h"

using aurora::models::normalize;

normalize::~normalize() {

}

normalize::normalize() {

}

normalize::normalize(size_t a_units) {
	units = a_units;
}

void normalize::pmt_wise(function<void(ptr<param>&)> a_func) {

}

model* normalize::clone() {
	normalize* result = new normalize();
	result->units = units;
	return result;
}

model* normalize::clone(function<void(ptr<param>&)> a_func) {
	normalize* result = new normalize();
	result->units = units;
	return result;
}

void normalize::fwd() {
	x.abs_1d(x_abs);
	sum = x_abs.sum_1d();
	assert(sum != 0);
	for (int i = 0; i < units; i++)
		y[i].val() = x[i] / sum;
}

void normalize::bwd() {
	double reciprocal = 1.0 / sum;
	double reciprocal_squared = reciprocal * reciprocal;
	for (int i = 0; i < units; i++) {
		x_grad[i].val() = y_grad[i] * (reciprocal - reciprocal_squared * x[i]);
		assert(x_grad[i] != 0 && !isnan(x_grad[i]) && !isinf(x_grad[i]));
	}
}

tensor& normalize::fwd(tensor& a_x) {
	x.pop(a_x);
	fwd();
	return y;
}

tensor& normalize::bwd(tensor& a_y_grad) {
	y_grad.pop(a_y_grad);
	bwd();
	return x_grad;
}

void normalize::signal(tensor& a_y_des) {
	y.sub_1d(a_y_des, y_grad);
}

void normalize::cycle(tensor& a_x, tensor& a_y_des) {
	x.pop(a_x);
	fwd();
	signal(a_y_des);
	bwd();
}

void normalize::recur(function<void(model*)> a_func) {
	a_func(this);
}

void normalize::compile() {
	x = tensor::new_1d(units);
	x_grad = tensor::new_1d(units);
	y = tensor::new_1d(units);
	y_grad = tensor::new_1d(units);
	x_abs = tensor::new_1d(units);
}

