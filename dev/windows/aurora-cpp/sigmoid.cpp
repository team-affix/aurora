#include "sigmoid.h"

using aurora::modeling::sigmoid;

sigmoid::~sigmoid() {

}

sigmoid::sigmoid() {

}

model* sigmoid::clone() {
	return new sigmoid();
}

model* sigmoid::clone(vector<param*>& a_pl) {
	return new sigmoid();
}

model* sigmoid::clone(vector<param_sgd*>& a_pl) {
	return new sigmoid();
}

model* sigmoid::clone(vector<param_mom*>& a_pl) {
	return new sigmoid();
}

void sigmoid::fwd() {
	y.val() = (double)1 / ((double)1 + exp(-x.val()));
}

void sigmoid::bwd() {
	x_grad.val() = y_grad.val() * y.val() * (1 - y.val());
}

tensor& sigmoid::fwd(tensor a_x) {
	x.set(a_x);
	fwd();
	return y;
}

tensor& sigmoid::bwd(tensor a_y_grad) {
	y_grad.set(a_y_grad);
	bwd();
	return x_grad;
}

void sigmoid::signal(tensor a_y_des) {
	y_grad.val() = y.val() - a_y_des.val();
}

void sigmoid::cycle(tensor a_x, tensor a_y_des) {
	x.set(a_x);
	fwd();
	signal(a_y_des);
	bwd();
}

void sigmoid::recur(function<void(model*)> a_func) {
	a_func(this);
}

void sigmoid::compile() {

}