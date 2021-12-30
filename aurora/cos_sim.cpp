#include "affix-base/pch.h"
#include "cos_sim.h"

using aurora::models::cos_sim;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::maths::tensor;

cos_sim::~cos_sim() {

}

cos_sim::cos_sim() {

}

cos_sim::cos_sim(size_t a_units) {
	m_units = a_units;
}

void cos_sim::param_recur(const function<void(Param&)>& a_func) {

}

model* cos_sim::clone(const function<Param(Param&)>& a_func) {
	cos_sim* result = new cos_sim();
	result->m_units = m_units;
	result->m_magnitude_0 = m_magnitude_0;
	result->m_magnitude_1 = m_magnitude_1;
	result->m_magnitude_product = m_magnitude_product;
	result->m_dot_product = m_dot_product;
	return result;
}

void cos_sim::fwd() {
	m_magnitude_0 = m_x[0].mag_1d();
	m_magnitude_1 = m_x[1].mag_1d();
	m_magnitude_product = m_magnitude_0 * m_magnitude_1;
	m_dot_product = m_x[0].dot_1d(m_x[1]);
	m_y.val() = m_dot_product / m_magnitude_product;
}

void cos_sim::bwd() {
	for (int i = 0; i < m_x.width(); i++) {
		double x_0_a = m_x[1][i];
		double x_0_b = m_x[0][i] * m_dot_product;
		double x_0_c = m_magnitude_product;
		double x_0_d = m_magnitude_1 * pow(m_magnitude_0, 3);
		m_x_grad[0][i].val() = m_y_grad * (x_0_a / x_0_c - x_0_b / x_0_d);
		double x_1_a = m_x[0][i];
		double x_1_b = m_x[1][i] * m_dot_product;
		double x_1_c = m_magnitude_product;
		double x_1_d = m_magnitude_0 * pow(m_magnitude_1, 3);
		m_x_grad[1][i].val() = m_y_grad * (x_1_a / x_1_c - x_1_b / x_1_d);
	}
}

void cos_sim::model_recur(const function<void(model*)>& a_func) {
	a_func(this);
}

void cos_sim::compile() {
	m_x = tensor::new_2d(2, m_units);
	m_x_grad = tensor::new_2d(2, m_units);
}
