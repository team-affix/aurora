#pragma once
#include "param.h"
#include "param_sgd.h"
#include "param_mom.h"

using aurora::optimization::param;
using aurora::optimization::param_sgd;
using aurora::optimization::param_mom;
using std::lock_guard;

param::~param() {

}

double& param::state() {
	lock_guard<mutex> lock(state_mtx);
	return state_ptr.val();
}

double& param::learn_rate() {
	lock_guard<mutex> lock(learn_rate_mtx);
	return learn_rate_ptr.val();
}

param* param::clone() {
	param* result = new param();
	result->state() = state();
	result->learn_rate() = learn_rate();
	return result;
}

param* param::to_param() {
	return clone();
}

param_sgd* param::to_param_sgd() {
	param_sgd* result = new param_sgd();
	result->state() = state();
	result->learn_rate() = learn_rate();
	return result;
}

param_mom* param::to_param_mom() {
	param_mom* result = new param_mom();
	result->state() = state();
	result->learn_rate() = learn_rate();
	return result;
}