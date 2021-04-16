#include "pch.h"
#include "tanh.h"

using aurora::models::tanh;

tanh::~tanh() {

}

tanh::tanh() {

}

void tanh::pmt_wise(function<void(ptr<param>&)> a_func) {

}

model* tanh::clone() {
	return new tanh();
}

model* tanh::clone(function<void(ptr<param>&)> a_init) {
	return new tanh();
}

void tanh::fwd() {
	y.val() = std::tanh(x.val());
}

void tanh::bwd() {
	x_grad.val() = y_grad.val() / pow(cosh(x.val()), 2);
}

tensor& tanh::fwd(tensor& a_x) {
	x.val() = a_x.val();
	fwd();
	return y;
}

tensor& tanh::bwd(tensor& a_y_grad) {
	y_grad.val() = a_y_grad.val();
	bwd();
	return x_grad;
}

void tanh::signal(tensor& a_y_des) {
	y_grad.val() = y.val() - a_y_des.val();
}

void tanh::cycle(tensor& a_x, tensor& a_y_des) {
	x.val() = a_x.val();
	fwd();
	signal(a_y_des);
	bwd();
}

void tanh::recur(function<void(model*)> a_func) {
	a_func(this);
}

void tanh::compile() {

}