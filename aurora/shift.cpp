#include "affix-base/pch.h"
#include "shift.h"

using aurora::models::shift;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::params::param_sgd;
using aurora::maths::tensor;
using std::vector;
using std::initializer_list;

shift::~shift() {

}

shift::shift() {

}

shift::shift(size_t a_units, vector<int> a_valid_shifts) {
	m_units = a_units;
	m_valid_shifts = a_valid_shifts;
}

void shift::param_recur(const function<void(Param&)>& a_func) {

}

model* shift::clone(const function<Param(Param&)>& a_func) {
	shift* result = new shift();
	result->m_units = m_units;
	result->m_valid_shifts = m_valid_shifts;
	return result;
}

void shift::fwd() {
	m_y.clear();
	for (int i = 0; i < m_units; i++)
		for (int j = 0; j < m_valid_shifts.size(); j++) {
			int dst = positive_modulo(i + m_valid_shifts[j], m_units);
			m_y[dst].val() += m_x[i] * m_amount[j];
		}
}

void shift::bwd() {
	m_x_grad.clear();
	m_amount_grad.clear();
	for (int i = 0; i < m_units; i++)
		for (int j = 0; j < m_valid_shifts.size(); j++) {
			int src = positive_modulo(i - m_valid_shifts[j], m_units);
			m_x_grad[src].val() += m_y_grad[i] * m_amount[j];
			m_amount_grad[j].val() += m_y_grad[i] * m_x[src];
		}
}

void shift::model_recur(const function<void(model*)>& a_func) {
	a_func(this);
}

void shift::compile() {
	m_x = tensor::new_1d(m_units);
	m_x_grad = tensor::new_1d(m_units);
	m_y = tensor::new_1d(m_units);
	m_y_grad = tensor::new_1d(m_units);
	m_amount = tensor::new_1d(m_valid_shifts.size());
	m_amount_grad = tensor::new_1d(m_valid_shifts.size());
}

int shift::positive_modulo(int i, int n) {
	return (i % n + n) % n;
}