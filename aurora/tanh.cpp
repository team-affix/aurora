#include "pch.h"
#include "tanh.h"

using aurora::models::tanh;

tanh::~tanh() {

}

tanh::tanh() {

}

tanh::tanh(double a_a, double a_b, double a_c) {
	a = a_a;
	b = a_b;
	c = a_c;
}

void tanh::param_recur(function<void(Param&)> a_func) {

}

model* tanh::clone(function<Param(Param&)> a_func) {
	return new tanh();
}

void tanh::fwd() {
	y.val() = a * std::tanh(b * x.val()) + c;
}

void tanh::bwd() {
	x_grad.val() = y_grad.val() * a / pow(cosh(b * x.val()), 2) * b;
}

void tanh::signal(const tensor& a_y_des) {
	y_grad.val() = y.val() - a_y_des.val();
}

void tanh::model_recur(function<void(model*)> a_func) {
	a_func(this);
}

void tanh::compile() {

}