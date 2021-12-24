#include "affix-base/pch.h"
#include "bias.h"
#include <iostream>

using aurora::models::bias;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::params::param_sgd;
using aurora::maths::tensor;

bias::~bias() {

}

bias::bias() {

}

void bias::param_recur(function<void(Param&)> a_func) {
	a_func(pmt);
}

model* bias::clone(function<Param(Param&)> a_func) {
	bias* result = new bias();
	result->pmt = a_func(pmt);
	return result;
}

void bias::fwd() {
	y.val() = x.val() + pmt->state();
}

void bias::bwd() {
	param_sgd* pmt_sgd = (param_sgd*)pmt.get();
	pmt_sgd->accum_grad(y_grad.val());
}

void bias::signal(const tensor& a_y_des) {
	y_grad.val() = y.val() - a_y_des.val();
}

void bias::model_recur(function<void(model*)> a_func) {
	a_func(this);
}

void bias::compile() {
	x_grad.group_add(y_grad);
}