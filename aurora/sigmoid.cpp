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

model* sigmoid::clone(function<void(ptr<param>&)> a_func) {
	return new sigmoid();
}

void sigmoid::fwd() {
	y.val() = (double)1 / ((double)1 + exp(-x.val()));
}

void sigmoid::bwd() {
	x_grad.val() = y_grad.val() * y.val() * (1 - y.val());
}

void sigmoid::signal(tensor& a_y_des) {
	y_grad.val() = y.val() - a_y_des.val();
}

void sigmoid::recur(function<void(model*)> a_func) {
	a_func(this);
}

void sigmoid::compile() {

}