#pragma once
#include "param.h"
#include "param_sgd.h"
#include "param_mom.h"

using aurora::optimization::param;
using aurora::optimization::param_sgd;
using aurora::optimization::param_mom;

param_mom::~param_mom() {

}

param_mom::param_mom() {

}

param_mom::param_mom(double a_state, double a_learn_rate, double a_gradient, double a_momentum, double a_beta) : param_sgd(a_state, a_learn_rate, a_gradient) {
	this->momentum() = a_momentum;
	this->beta() = a_beta;
}

double& param_mom::momentum() {
	return momentum_ptr.val();
}

double& param_mom::beta() {
	return beta_ptr.val();
}

param* param_mom::clone() {
	param_mom* result = new param_mom();
	result->state() = state();
	result->learn_rate() = learn_rate();
	result->gradient() = gradient();
	result->momentum() = momentum();
	result->beta() = beta();
	return result;
}

param* param_mom::to_param() {
	param* result = new param();
	result->state() = state();
	result->learn_rate() = learn_rate();
	return result;
}

param_sgd* param_mom::to_param_sgd() {
	param_sgd* result = new param_sgd();
	result->state() = state();
	result->learn_rate() = learn_rate();
	result->gradient() = gradient();
	return result;
}

param_mom* param_mom::to_param_mom() {
	return (param_mom*)clone();
}