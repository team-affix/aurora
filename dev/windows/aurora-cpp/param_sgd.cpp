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

void param_sgd::accum_grad(double a_grad) {
	gradient() += a_grad;
}

param* param_sgd::clone() {
	param_sgd* result = new param_sgd();
	result->state() = state();
	result->learn_rate() = learn_rate();
	result->gradient() = gradient();
	return result;
}