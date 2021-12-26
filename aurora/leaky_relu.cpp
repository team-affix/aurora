#include "affix-base/pch.h"
#include "leaky_relu.h"

using aurora::models::leaky_relu;
using std::function;
using aurora::models::model;
using aurora::params::Param;
using aurora::maths::tensor;

leaky_relu::~leaky_relu() {

}

leaky_relu::leaky_relu(double a_m) {
	m_m.val() = a_m;
}

void leaky_relu::param_recur(function<void(Param&)> a_func) {

}

model* leaky_relu::clone(function<Param(Param&)> a_func) {
	return new leaky_relu(m_m.val());
}

void leaky_relu::fwd() {
	if (m_x.val() > 0)
		m_y.val() = m_x.val();
	else
		m_y.val() = m_m.val() * m_x.val();
}

void leaky_relu::bwd() {
	if (m_x.val() > 0)
		m_x_grad.val() = m_y_grad.val();
	else
		m_x_grad.val() = m_y_grad.val() * m_m.val();
}

void leaky_relu::signal(const tensor& a_y_des) {
	m_y_grad.val() = m_y.val() - a_y_des.val();
}

void leaky_relu::model_recur(function<void(model*)> a_func) {
	a_func(this);
}

void leaky_relu::compile() {

}
