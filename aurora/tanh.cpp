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

void tanh::pmt_wise(function<void(ptr<param>&)> a_func) {

}

model* tanh::clone() {
	return new tanh();
}

model* tanh::clone(function<void(ptr<param>&)> a_func) {
	return new tanh();
}

void tanh::fwd() {
	y.val() = a * std::tanh(b * x.val()) + c;
}

void tanh::bwd() {
	x_grad.val() = y_grad.val() * a / pow(cosh(b * x.val()), 2) * b;
}

void tanh::signal(tensor& a_y_des) {
	y_grad.val() = y.val() - a_y_des.val();
}

void tanh::recur(function<void(model*)> a_func) {
	a_func(this);
}

void tanh::compile() {

}