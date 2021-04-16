#include "pch.h"
#include "sigmoid.h"

using aurora::models::sigmoid;

sigmoid::~sigmoid() {

}

sigmoid::sigmoid() {

}

void sigmoid::pmt_wise(function<void(ptr<param>&)> a_func) {

}

model* sigmoid::clone() {
	return new sigmoid();
}

model* sigmoid::clone(function<void(ptr<param>&)> a_init) {
	return new sigmoid();
}

void sigmoid::fwd() {
	y.val() = (double)1 / ((double)1 + exp(-x.val()));
}

void sigmoid::bwd() {
	x_grad.val() = y_grad.val() * y.val() * (1 - y.val());
}

tensor& sigmoid::fwd(tensor& a_x) {
	x.val() = a_x.val();
	fwd();
	return y;
}

tensor& sigmoid::bwd(tensor& a_y_grad) {
	y_grad.val() = a_y_grad.val();
	bwd();
	return x_grad;
}

void sigmoid::signal(tensor& a_y_des) {
	y_grad.val() = y.val() - a_y_des.val();
}

void sigmoid::cycle(tensor& a_x, tensor& a_y_des) {
	x.val() = a_x.val();
	fwd();
	signal(a_y_des);
	bwd();
}

void sigmoid::recur(function<void(model*)> a_func) {
	a_func(this);
}

void sigmoid::compile() {

}