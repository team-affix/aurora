#include "affix-base/pch.h"
#include "param_rcv.h"

using aurora::params::param;
using aurora::params::param_rcv;

param_rcv::~param_rcv() {

}

param_rcv::param_rcv() {

}

param_rcv::param_rcv(const double& a_learn_rate, const double& a_beta)
{
	learn_rate() = a_learn_rate;
	beta(a_beta);
}

double param_rcv::sign(const double& a_x) {
	if (a_x >= 0)
		return 1;
	else
		return -1;
}

double& param_rcv::dstate() {
	return m_dstate.val();
}

double& param_rcv::reward() {
	return m_reward.val();
}

double& param_rcv::dreward() {
	return m_dreward.val();
}

double& param_rcv::learn_rate() {
	return m_learn_rate.val();
}

const double& param_rcv::learn_rate() const {
	return m_learn_rate.val();
}

const double& param_rcv::beta() const
{
	return m_beta.val();
}

double& param_rcv::momentum() {
	return m_momentum.val();
}

const double& param_rcv::alpha() {
	return m_alpha.val();
}

void param_rcv::beta(const double& a_val) {
	assert(a_val >= 0 && a_val <= 1);
	m_beta.val() = a_val;
	m_alpha.val() = 1.0 - a_val;
}

void param_rcv::update(const double& a_c) {
	dstate() = learn_rate() * (beta() * momentum() + alpha() * a_c);
	state() += dstate();
}

void param_rcv::reward(const double& a_reward) {
	dreward() = a_reward - reward();
	reward() = a_reward;
	double slope = (dreward() / dstate());
	momentum() = beta() * momentum() + alpha() * sign(slope);
}

void param_rcv::dreward(const double& a_dreward) {
	dreward() = a_dreward;
	reward() += a_dreward;
	double slope = (dreward() / dstate());
	momentum() = beta() * momentum() + alpha() * sign(slope);
}

param* param_rcv::clone() const
{
	param_rcv* result = new param_rcv();
	result->m_dstate.val() = m_dstate.val();
	result->m_learn_rate.val() = m_learn_rate.val();
	result->m_reward.val() = m_reward.val();
	result->m_dreward.val() = m_dreward.val();
	result->m_beta.val() = m_beta.val();
	result->m_alpha.val() = m_alpha.val();
	result->m_momentum.val() = m_momentum.val();
	return result;
}
