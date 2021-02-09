#include "param_sgd_mt.h"

using aurora::optimization::param_sgd_mt;

param_sgd_mt::~param_sgd_mt() {

}

param_sgd_mt::param_sgd_mt() {

}

param_sgd_mt::param_sgd_mt(double a_state, double a_learn_rate, double a_gradient) {
	this->state() = a_state;
	this->learn_rate() = a_learn_rate;
	this->gradient() = a_gradient;
}

double& param_sgd_mt::state() {
	lock_guard<mutex> lock(state_mtx);
	return state_ptr.val();
}

double& param_sgd_mt::learn_rate() {
	lock_guard<mutex> lock(learn_rate_mtx);
	return learn_rate_ptr.val();
}

double& param_sgd_mt::gradient() {
	lock_guard<mutex> lock(gradient_mtx);
	return gradient_ptr.val();
}

param* param_sgd_mt::clone() {
	param_sgd_mt* result = new param_sgd_mt();
	result->state() = state();
	result->learn_rate() = learn_rate();
	result->gradient() = gradient();
	return result;
}