#include "affix-base/pch.h"
#include "param_mom_mt.h"

using aurora::params::param;
using aurora::params::param_mom_mt;

param_mom_mt::~param_mom_mt() {

}

param_mom_mt::param_mom_mt() {

}

param_mom_mt::param_mom_mt(double a_state, double a_learn_rate, double a_gradient, double a_momentum, double a_beta) : param_mom(a_state, a_learn_rate, a_gradient, a_momentum, a_beta) {

}

void param_mom_mt::accum_grad(double a_grad) {
	m_accum_grad_mtx.lock();
	gradient() += a_grad;
	m_accum_grad_mtx.unlock();
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
