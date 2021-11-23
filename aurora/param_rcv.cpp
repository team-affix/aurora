#include "pch.h"
#include "param_rcv.h"
#include "static_vals.h"

using aurora::params::param_rcv;

uniform_real_distribution<double> param_rcv::m_percentage_urd(0, 1);
normal_distribution<double> param_rcv::m_rcv_nd(1, 1);
uniform_real_distribution<double> param_rcv::m_rcv_urd(0.99, 1.01);

param_rcv::~param_rcv() {

}

param_rcv::param_rcv() {

}

param_rcv::param_rcv(const double& a_state, const double& a_learn_rate, const double& a_beta) {
	state() = a_state;
	learn_rate() = a_learn_rate;
	beta(a_beta);
}

double& param_rcv::learn_rate() {
	return m_learn_rate.val();
}

const double& param_rcv::beta() {
	return m_beta.val();
}

const double& param_rcv::alpha() {
	return m_alpha.val();
}

double& param_rcv::running_average() {
	return m_running_average.val();
}

double param_rcv::sign(const double& a_x) {
	if (a_x >= 0)
		return 1.0;
	else
		return -1.0;
}

void param_rcv::beta(const double& a_val) {
	assert(a_val >= 0 && a_val <= 1);
	m_beta.val() = a_val;
	m_alpha.val() = 1.0 - a_val;
}

void param_rcv::update(const double& a_loss) {

	double dl = a_loss - m_l_prev.val();
	double ds = state() - m_s_prev.val();

	// CALCULATE NEW DL/DS
	running_average() = beta() * running_average() + alpha() * (dl / ds);
	
	// SAVE CURRENT VALUES OF LOSS AND STATE
	m_s_prev.val() = state();
	m_l_prev.val() = a_loss;

	// UPDATE STATE
	state() -= learn_rate() * m_rcv_urd(static_vals::random_engine) * sign(running_average());
	/*if (m_percentage_urd(static_vals::random_engine) > 0.001) {
	}*/

}
