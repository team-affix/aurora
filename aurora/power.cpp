#include "affix-base/pch.h"
#include "power.h"

using aurora::models::power;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::params::param_sgd;
using aurora::maths::tensor;
using std::vector;

power::~power() {

}

power::power() {

}

power::power(size_t a_units) {
	m_units = a_units;
}

void power::param_recur(const function<void(Param&)>& a_func) {

}

model* power::clone(const function<Param(Param&)>& a_func) {
	power* result = new power();
	result->m_units = m_units;
	return result;
}

void power::fwd() {
	double& pow_amount = m_amount[0];
	for (int i = 0; i < m_units; i++) {
		m_y[i].val() = pow(m_x[i], pow_amount);
	}
}

void power::bwd() {
	double& pow_amount = m_amount[0];
	double& pow_amount_grad = m_amount_grad[0];
	pow_amount_grad = 0;
	for (int i = 0; i < m_units; i++) {
		m_x_grad[i].val() = m_y_grad[i] * pow_amount * pow(m_x[i], pow_amount - 1.0);
		pow_amount_grad += m_y_grad[i] * log(m_x[i]) * m_y[i];
	}
}

void power::model_recur(const function<void(model*)>& a_func) {
	a_func(this);
}

void power::compile() {
	m_x = tensor::new_1d(m_units);
	m_x_grad = tensor::new_1d(m_units);
	m_y = tensor::new_1d(m_units);
	m_y_grad = tensor::new_1d(m_units);
	m_amount = tensor::new_1d(1);
	m_amount_grad = tensor::new_1d(1);
}
