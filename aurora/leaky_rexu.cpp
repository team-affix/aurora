#include "affix-base/pch.h"
#include "leaky_rexu.h"

using aurora::models::leaky_rexu;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::params::param_sgd;
using aurora::maths::tensor;

leaky_rexu::~leaky_rexu() {

}

leaky_rexu::leaky_rexu() {

}

leaky_rexu::leaky_rexu(double a_k) {
	m_k = a_k;
}

void leaky_rexu::param_recur(const function<void(Param&)>& a_func) {

}

model* leaky_rexu::clone(const function<Param(Param&)>& a_func) {
	leaky_rexu* result = new leaky_rexu();
	result->m_k = m_k;
	return result;
}

void leaky_rexu::fwd() {
	if (m_x < 0)
		m_y.val() = exp(m_x) + m_k_minus_one;
	else
		m_y.val() = m_x + m_k;
}

void leaky_rexu::bwd() {
	if (m_x < 0)
		m_x_grad.val() = m_y_grad * (m_y - m_k_minus_one);
	else
		m_x_grad.val() = m_y_grad;
}

void leaky_rexu::model_recur(const function<void(model*)>& a_func) {
	a_func(this);
}

void leaky_rexu::compile() {
	m_k_minus_one = m_k - 1.0;
}