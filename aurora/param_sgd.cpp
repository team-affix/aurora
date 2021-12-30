#include "affix-base/pch.h"
#include "param.h"
#include "param_sgd.h"
#include "param_mom.h"

using aurora::params::param;
using aurora::params::param_sgd;
using aurora::params::param_mom;

param_sgd::~param_sgd() {

}

param_sgd::param_sgd() {

}

param_sgd::param_sgd(
	double a_learn_rate
)
{
	this->learn_rate() = a_learn_rate;
}

double& param_sgd::learn_rate() {
	return m_learn_rate_ptr.val();
}

const double& param_sgd::learn_rate() const
{
	return m_learn_rate_ptr.val();
}

double& param_sgd::gradient() {
	return m_gradient_ptr.val();
}

const double& param_sgd::gradient() const
{
	return m_gradient_ptr.val();
}

void param_sgd::accum_grad(double a_grad) {
	gradient() += a_grad;
}

void param_sgd::update() {
	assert(!isnan(state()) && !isnan(learn_rate()) && !isnan(gradient()));
	state() -= learn_rate() * gradient();
	gradient() = 0;
}

param* param_sgd::clone() const
{
	param_sgd* result = new param_sgd();
	result->state() = state();
	result->learn_rate() = learn_rate();
	result->gradient() = gradient();
	return result;
}
