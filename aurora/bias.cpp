#include "affix-base/pch.h"
#include "bias.h"
#include <iostream>

using aurora::models::bias;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::params::param_sgd;
using aurora::maths::tensor;

bias::~bias() {

}

bias::bias() {

}

void bias::param_recur(const function<void(Param&)>& a_func) {
	a_func(m_param);
}

model* bias::clone(const function<Param(Param&)>& a_func) {
	bias* result = new bias();
	result->m_param = a_func(m_param);
	return result;
}

void bias::fwd() {
	m_y.val() = m_x.val() + m_param->state();
}

void bias::bwd() {
	param_sgd* pmt_sgd = (param_sgd*)m_param.get();
	pmt_sgd->accum_grad(m_y_grad.val());
}

void bias::signal(const tensor& a_y_des) {
	m_y_grad.val() = m_y.val() - a_y_des.val();
}

void bias::model_recur(const function<void(model*)>& a_func) {
	a_func(this);
}

void bias::compile() {
	m_x_grad.group_add(m_y_grad);
}