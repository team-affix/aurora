#include "affix-base/pch.h"
#include "param.h"
#include "param_sgd.h"
#include "param_mom.h"

using aurora::params::param;
using aurora::params::param_sgd;
using aurora::params::param_mom;

param_mom::~param_mom() {

}

param_mom::param_mom() {

}

param_mom::param_mom(
	double a_learn_rate,
	double a_beta
) :
	param_sgd(a_learn_rate)
{
	this->beta() = a_beta;
}

double& param_mom::momentum() {
	return m_momentum_ptr.val();
}

const double& param_mom::momentum() const
{
	return m_momentum_ptr.val();
}

double& param_mom::beta() {
	return m_beta_ptr.val();
}

const double& param_mom::beta() const
{
	return m_beta_ptr.val();
}

void param_mom::update() {
	momentum() = beta() * momentum() + (1 - beta()) * gradient();
	state() -= learn_rate() * momentum();
	gradient() = 0;
}

param* param_mom::clone() const
{
	param_mom* result = new param_mom();
	result->state() = state();
	result->learn_rate() = learn_rate();
	result->gradient() = gradient();
	result->momentum() = momentum();
	result->beta() = beta();
	return result;
}
