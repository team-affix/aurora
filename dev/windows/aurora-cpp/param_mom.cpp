#pragma once
#include "param.h"
#include "param_sgd.h"
#include "param_mom.h"

using aurora::optimization::param;
using aurora::optimization::param_sgd;
using aurora::optimization::param_mom;

param_mom::~param_mom() {

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