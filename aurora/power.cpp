#include "pch.h"
#include "power.h"

using aurora::models::power;

power::~power() {

}

power::power() {

}

power::power(size_t a_units) {
	units = a_units;
}

void power::pmt_wise(function<void(ptr<param>&)> a_func) {

}

model* power::clone() {
	power* result = new power();
	result->units = units;
	return result;
}

model* power::clone(function<void(ptr<param>&)> a_func) {
	power* result = new power();
	result->units = units;
	return result;
}

void power::fwd() {
	double& pow_amount = amount[0];
	for (int i = 0; i < units; i++) {
		y[i].val() = pow(x[i], pow_amount);
	}
}

void power::bwd() {
	double& pow_amount = amount[0];
	double& pow_amount_grad = amount_grad[0];
	pow_amount_grad = 0;
	for (int i = 0; i < units; i++) {
		x_grad[i].val() = y_grad[i] * pow_amount * pow(x[i], pow_amount - 1.0);
		pow_amount_grad += y_grad[i] * log(x[i]) * y[i];
	}
}

tensor& power::fwd(tensor& a_x) {
	x.pop(a_x);
	fwd();
	return y;
}

tensor& power::bwd(tensor& a_y_grad) {
	y_grad.pop(a_y_grad);
	bwd();
	return x_grad;
}

void power::signal(tensor& a_y_des) {
	y.sub_1d(a_y_des, y_grad);
}

void power::cycle(tensor& a_x, tensor& a_y_des) {
	x.pop(a_x);
	fwd();
	signal(a_y_des);
	bwd();
}

void power::recur(function<void(model*)> a_func) {
	a_func(this);
}

void power::compile() {
	x = tensor::new_1d(units);
	x_grad = tensor::new_1d(units);
	y = tensor::new_1d(units);
	y_grad = tensor::new_1d(units);
	amount = tensor::new_1d(1);
	amount_grad = tensor::new_1d(1);
}