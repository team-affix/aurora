#include "affix-base/pch.h"
#include "sigmoid.h"

using aurora::models::sigmoid;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::params::param_sgd;
using aurora::maths::tensor;
using std::vector;
using std::initializer_list;

sigmoid::~sigmoid() {

}

sigmoid::sigmoid() {

}

void sigmoid::param_recur(function<void(Param&)> a_func) {

}

model* sigmoid::clone(function<Param(Param&)> a_func) {
	return new sigmoid();
}

void sigmoid::fwd() {
	y.val() = (double)1 / ((double)1 + exp(-x.val()));
}

void sigmoid::bwd() {
	x_grad.val() = y_grad.val() * y.val() * (1 - y.val());
}

void sigmoid::signal(const tensor& a_y_des) {
	y_grad.val() = y.val() - a_y_des.val();
}

void sigmoid::model_recur(function<void(model*)> a_func) {
	a_func(this);
}

void sigmoid::compile() {

}