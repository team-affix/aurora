#include "affix-base/pch.h"
#include "interpolate.h"

using aurora::models::interpolate;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::maths::tensor;

interpolate::~interpolate() {

}

interpolate::interpolate() {

}

interpolate::interpolate(size_t a_units) {
	m_units = a_units;
}

void interpolate::param_recur(function<void(Param&)> a_func) {

}

model* interpolate::clone(function<Param(Param&)> a_func) {
	interpolate* result = new interpolate();
	result->m_units = m_units;
	return result;
}

void interpolate::fwd() {
	double& interpolate_amount = m_amount[0];
	m_amount_compliment = 1.0 - interpolate_amount;
	for (int i = 0; i < m_units; i++)
		m_y[i].val() = m_x[1][i] * interpolate_amount + m_x[0][i] * m_amount_compliment;
}

void interpolate::bwd() {
	double& interpolate_amount = m_amount[0];
	double& interpolate_amount_grad = m_amount_grad[0];
	interpolate_amount_grad = 0;
	for (int i = 0; i < m_units; i++) {
		m_x_grad[1][i].val() = m_y_grad[i] * interpolate_amount;
		m_x_grad[0][i].val() = m_y_grad[i] * m_amount_compliment;
		interpolate_amount_grad += m_y_grad[i] * (m_x[1][i] - m_x[0][i]);
	}
}

void interpolate::signal(const tensor& a_y_des) {
	m_y.sub_1d(a_y_des, m_y_grad);
}

void interpolate::model_recur(function<void(model*)> a_func) {
	a_func(this);
}

void interpolate::compile() {
	m_x = tensor::new_2d(2, m_units);
	m_x_grad = tensor::new_2d(2, m_units);
	m_y = tensor::new_1d(m_units);
	m_y_grad = tensor::new_1d(m_units);
	m_amount = tensor::new_1d(1);
	m_amount_grad = tensor::new_1d(1);
}
