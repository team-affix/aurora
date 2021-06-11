#include "pch.h"
#include "interpolate.h"

using aurora::models::interpolate;

interpolate::~interpolate() {

}

interpolate::interpolate() {

}

interpolate::interpolate(size_t a_units) {
	units = a_units;
}

void interpolate::param_recur(function<void(ptr<param>&)> a_func) {

}

model* interpolate::clone() {
	interpolate* result = new interpolate();
	result->units = units;
	return result;
}

model* interpolate::clone(function<void(ptr<param>&)> a_func) {
	interpolate* result = new interpolate();
	result->units = units;
	return result;
}

void interpolate::fwd() {
	double& interpolate_amount = amount[0];
	amount_compliment = 1.0 - interpolate_amount;
	for (int i = 0; i < units; i++)
		y[i].val() = x[1][i] * interpolate_amount + x[0][i] * amount_compliment;
}

void interpolate::bwd() {
	double& interpolate_amount = amount[0];
	double& interpolate_amount_grad = amount_grad[0];
	interpolate_amount_grad = 0;
	for (int i = 0; i < units; i++) {
		x_grad[1][i].val() = y_grad[i] * interpolate_amount;
		x_grad[0][i].val() = y_grad[i] * amount_compliment;
		interpolate_amount_grad += y_grad[i] * (x[1][i] - x[0][i]);
	}
}

void interpolate::signal(tensor& a_y_des) {
	y.sub_1d(a_y_des, y_grad);
}

void interpolate::model_recur(function<void(model*)> a_func) {
	a_func(this);
}

void interpolate::compile() {
	x = tensor::new_2d(2, units);
	x_grad = tensor::new_2d(2, units);
	y = tensor::new_1d(units);
	y_grad = tensor::new_1d(units);
	amount = tensor::new_1d(1);
	amount_grad = tensor::new_1d(1);
}