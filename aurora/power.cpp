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

void power::param_recur(function<void(Param&)> a_func) {

}

model* power::clone(function<Param(Param&)> a_func) {
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

void power::signal(tensor& a_y_des) {
	y.sub_1d(a_y_des, y_grad);
}

void power::model_recur(function<void(model*)> a_func) {
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