#pragma once
#include "param.h"
#include "param_sgd.h"
#include "param_mom.h"

using aurora::optimization::param;
using aurora::optimization::param_sgd;
using aurora::optimization::param_mom;

param_sgd::~param_sgd() {

}

param_sgd::param_sgd() {

}

param_sgd::param_sgd(double a_state, double a_learn_rate, double a_gradient) : param(a_state, a_learn_rate) {
	this->gradient() = a_gradient;
}

double& param_sgd::gradient() {
	return gradient_ptr.val();
}

param* param_sgd::clone() {
	param_sgd* result = new param_sgd();
	result->state() = state();
	result->learn_rate() = learn_rate();
	result->gradient() = gradient();
	return result;
}

param* param_sgd::to_param() {
	param* result = new param();
	result->state() = state();
	result->learn_rate() = learn_rate();
	return result;
}

param_sgd* param_sgd::to_param_sgd() {
	return (param_sgd*)clone();
}

param_mom* param_sgd::to_param_mom() {
	param_mom* result = new param_mom();
	result->state() = state();
	result->learn_rate() = learn_rate();
	result->gradient() = gradient();
	return result;
}