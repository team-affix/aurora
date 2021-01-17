#include "tanh.h"

using aurora::modeling::tanh;

tanh::~tanh() {

}

tanh::tanh() {

}

model* tanh::clone() {
	return new tanh();
}

model* tanh::clone(vector<param*>& a_pl) {
	return new tanh();
}

model* tanh::clone(vector<param_sgd*>& a_pl) {
	return new tanh();
}

model* tanh::clone(vector<param_mom*>& a_pl) {
	return new tanh();
}

void tanh::fwd() {
	y.val() = std::tanh(x.val());
}

void tanh::bwd() {
	x_grad.val() = y_grad.val() / pow(cosh(x.val()), 2);
}

tensor& tanh::fwd(tensor a_x) {
	x.set(a_x);
	fwd();
	return y;
}

tensor& tanh::bwd(tensor a_y_grad) {
	y_grad.set(a_y_grad);
	bwd();
	return x_grad;
}

void tanh::signal(tensor a_y_des) {
	y_grad.val() = y.val() - a_y_des.val();
}

void tanh::cycle(tensor a_x, tensor a_y_des) {
	x.set(a_x);
	fwd();
	signal(a_y_des);
	bwd();
}

void tanh::recur(function<void(model*)> a_func) {
	a_func(this);
}

void tanh::compile() {

}