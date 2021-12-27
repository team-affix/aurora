#include "affix-base/pch.h"
#include "normalize.h"

using aurora::models::normalize;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::params::param_sgd;
using aurora::maths::tensor;
using std::vector;

normalize::~normalize() {

}

normalize::normalize() {

}

normalize::normalize(size_t a_units) {
	m_units = a_units;
}

void normalize::param_recur(const function<void(Param&)>& a_func) {

}

model* normalize::clone(const function<Param(Param&)>& a_func) {
	normalize* result = new normalize();
	result->m_units = m_units;
	return result;
}

void normalize::fwd() {
	m_x.abs_1d(m_x_abs);
	m_sum = m_x_abs.sum_1d();
	assert(m_sum != 0);
	for (int i = 0; i < m_units; i++)
		m_y[i].val() = m_x[i] / m_sum;
}

void normalize::bwd() {
	double reciprocal = 1.0 / m_sum;
	double reciprocal_squared = reciprocal * reciprocal;
	for (int i = 0; i < m_units; i++) {
		m_x_grad[i].val() = m_y_grad[i] * (reciprocal - reciprocal_squared * m_x[i]);
		//assert(x_grad[i] != 0 && !isnan(x_grad[i]) && !isinf(x_grad[i]));
	}
}

void normalize::signal(const tensor& a_y_des) {
	m_y.sub_1d(a_y_des, m_y_grad);
}

void normalize::model_recur(const function<void(model*)>& a_func) {
	a_func(this);
}

void normalize::compile() {
	m_x = tensor::new_1d(m_units);
	m_x_grad = tensor::new_1d(m_units);
	m_y = tensor::new_1d(m_units);
	m_y_grad = tensor::new_1d(m_units);
	m_x_abs = tensor::new_1d(m_units);
}

