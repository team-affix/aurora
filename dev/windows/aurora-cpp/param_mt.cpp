#include "param_mt.h"

using aurora::optimization::param_mt;

param_mt::~param_mt() {

}

param_mt::param_mt() {

}

param_mt::param_mt(double a_state, double a_learn_rate) {
	this->state() = a_state;
	this->learn_rate() = a_learn_rate;
}

double& param_mt::state() {
	lock_guard<mutex> lock(state_mtx);
	return state_ptr.val();
}

double& param_mt::learn_rate() {
	lock_guard<mutex> lock(learn_rate_mtx);
	return learn_rate_ptr.val();
}

param* param_mt::clone() {
	param_mt* result = new param_mt();
	result->state() = state();
	result->learn_rate() = learn_rate();
	return result;
}