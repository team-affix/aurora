#include "pch.h"
#include "shift.h"

using aurora::models::shift;

shift::~shift() {

}

shift::shift() {

}

shift::shift(size_t a_units, vector<int> a_valid_shifts) {
	units = a_units;
	valid_shifts = a_valid_shifts;
}

void shift::pmt_wise(function<void(ptr<param>&)> a_func) {

}

model* shift::clone() {
	shift* result = new shift();
	result->units = units;
	result->valid_shifts = valid_shifts;
	return result;
}

model* shift::clone(function<void(ptr<param>&)> a_func) {
	shift* result = new shift();
	result->units = units;
	result->valid_shifts = valid_shifts;
	return result;
}

void shift::fwd() {
	y.clear();
	for (int i = 0; i < units; i++)
		for (int j = 0; j < valid_shifts.size(); j++) {
			int dst = positive_modulo(i + valid_shifts[j], units);
			y[dst].val() += x[i] * amount[j];
		}
}

void shift::bwd() {
	x_grad.clear();
	amount_grad.clear();
	for (int i = 0; i < units; i++)
		for (int j = 0; j < valid_shifts.size(); j++) {
			int src = positive_modulo(i - valid_shifts[j], units);
			x_grad[src].val() += y_grad[i] * amount[j];
			amount_grad[j].val() += y_grad[i] * x[src];
		}
}

void shift::signal(tensor& a_y_des) {
	y.sub_1d(a_y_des, y_grad);
}

void shift::recur(function<void(model*)> a_func) {
	a_func(this);
}

void shift::compile() {
	x = tensor::new_1d(units);
	x_grad = tensor::new_1d(units);
	y = tensor::new_1d(units);
	y_grad = tensor::new_1d(units);
	amount = tensor::new_1d(valid_shifts.size());
	amount_grad = tensor::new_1d(valid_shifts.size());
}

int shift::positive_modulo(int i, int n) {
	return (i % n + n) % n;
}