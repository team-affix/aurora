#include "affix-base/pch.h"
#include "param.h"
#include "param_sgd.h"
#include "param_mom.h"

using aurora::params::param;
using aurora::params::param_sgd;
using aurora::params::param_mom;

param::~param() {

}

param::param() {

}

param::param(double a_state) {
	this->state() = a_state;
}

double& param::state() {
	return m_state_ptr.val();
}

const double& param::state() const
{
	return m_state_ptr.val();
}

void param::update() {

}

param* param::clone() const
{
	param* result = new param();
	result->state() = state();
	return result;
}
