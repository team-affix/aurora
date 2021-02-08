#pragma once
#include "param.h"
#include "param_sgd.h"
#include "param_mom.h"

using aurora::optimization::param;
using aurora::optimization::param_sgd;
using aurora::optimization::param_mom;

param::~param() {

}

param::param() {

}

param::param(double a_state, double a_learn_rate) {
	this->state() = a_state;
	this->learn_rate() = a_learn_rate;
}

double& param::state() {
	return state_ptr.val();
}

double& param::learn_rate() {
	return learn_rate_ptr.val();
}

param* param::clone() {
	param* result = new param();
	result->state() = state();
	result->learn_rate() = learn_rate();
	return result;
}