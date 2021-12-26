#include "affix-base/pch.h"
#include "ntm_sparsify.h"

using aurora::models::ntm_sparsify;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::params::param_sgd;
using aurora::maths::tensor;
using std::vector;

ntm_sparsify::~ntm_sparsify() {

}

ntm_sparsify::ntm_sparsify() {

}

ntm_sparsify::ntm_sparsify(size_t a_memory_height) {
	m_memory_height = a_memory_height;
}

void ntm_sparsify::param_recur(function<void(Param&)> a_func) {

}

model* ntm_sparsify::clone(function<Param(Param&)> a_func) {
	ntm_sparsify* result = new ntm_sparsify();
	result->m_memory_height = m_memory_height;
	return result;
}

void ntm_sparsify::fwd() {
	for (int i = 0; i < m_memory_height; i++)
		m_y[i].val() = exp(m_beta[0] * m_x[i].val());
}

void ntm_sparsify::bwd() {
	m_beta_grad[0].val() = 0;
	for (int i = 0; i < m_memory_height; i++) {
		m_x_grad[i].val() = m_y_grad[i] * m_y[i] * m_beta[0];
		m_beta_grad[0].val() += m_y_grad[i] * m_y[i] * m_x[i];
	}
}

void ntm_sparsify::signal(const tensor& a_y_des) {
	m_y.sub_1d(a_y_des, m_y_grad);
}

void ntm_sparsify::model_recur(function<void(model*)> a_func) {
	a_func(this);
}

void ntm_sparsify::compile() {
	m_x = tensor::new_1d(m_memory_height);
	m_x_grad = tensor::new_1d(m_memory_height);
	m_y = tensor::new_1d(m_memory_height);
	m_y_grad = tensor::new_1d(m_memory_height);
	m_beta = tensor::new_1d(1);
	m_beta_grad = tensor::new_1d(1);
}