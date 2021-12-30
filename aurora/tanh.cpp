#include "affix-base/pch.h"
#include "tanh.h"

using aurora::models::tanh;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::params::param_sgd;
using aurora::maths::tensor;
using std::vector;
using std::initializer_list;
using std::uniform_real_distribution;

tanh::~tanh() {

}

tanh::tanh() {

}

tanh::tanh(double a_a, double a_b, double a_c) {
	m_a = a_a;
	m_b = a_b;
	m_c = a_c;
}

void tanh::param_recur(const function<void(Param&)>& a_func) {

}

model* tanh::clone(const function<Param(Param&)>& a_func) {
	return new tanh();
}

void tanh::fwd() {
	m_y.val() = m_a * std::tanh(m_b * m_x.val()) + m_c;
}

void tanh::bwd() {
	m_x_grad.val() = m_y_grad.val() * m_a / pow(cosh(m_b * m_x.val()), 2) * m_b;
}

void tanh::model_recur(const function<void(model*)>& a_func) {
	a_func(this);
}

void tanh::compile() {

}
