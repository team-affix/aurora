#include "param_mom_mt.h"

using aurora::optimization::param_mom_mt;

param_mom_mt::~param_mom_mt() {

}

param_mom_mt::param_mom_mt() {

}

param_mom_mt::param_mom_mt(double a_state, double a_learn_rate, double a_gradient, double a_momentum, double a_beta) {
	this->state() = a_state;
	this->learn_rate() = a_learn_rate;
	this->gradient() = a_gradient;
	this->momentum() = a_momentum;
	this->beta() = a_beta;
}

double& param_mom_mt::state() {
	lock_guard<mutex> lock(state_mtx);
	return state_ptr.val();
}

double& param_mom_mt::learn_rate() {
	lock_guard<mutex> lock(learn_rate_mtx);
	return learn_rate_ptr.val();
}

double& param_mom_mt::gradient() {
	lock_guard<mutex> lock(gradient_mtx);
	return gradient_ptr.val();
}

double& param_mom_mt::momentum() {
	lock_guard<mutex> lock(momentum_mtx);
	return momentum_ptr.val();
}

double& param_mom_mt::beta() {
	lock_guard<mutex> lock(beta_mtx);
	return beta_ptr.val();
}

param* param_mom_mt::clone() {
	param_mom_mt* result = new param_mom_mt();
	result->state() = state();
	result->learn_rate() = learn_rate();
	result->gradient() = gradient();
	result->momentum() = momentum();
	result->beta() = beta();
	return result;
}