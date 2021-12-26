#include "affix-base/pch.h"
#include "param_sgd_mt.h"

using aurora::params::param;
using aurora::params::param_sgd_mt;

param_sgd_mt::~param_sgd_mt() {

}

param_sgd_mt::param_sgd_mt() {

}

param_sgd_mt::param_sgd_mt(double a_state, double a_learn_rate, double a_gradient) : param_sgd(a_state, a_learn_rate, a_gradient) {

}

void param_sgd_mt::accum_grad(double a_grad) {
	m_accum_grad_mutex.lock();
	gradient() += a_grad;
	m_accum_grad_mutex.unlock();
}

param* param_sgd_mt::clone() {
	param_sgd_mt* result = new param_sgd_mt();
	result->state() = state();
	result->learn_rate() = learn_rate();
	result->gradient() = gradient();
	return result;
}