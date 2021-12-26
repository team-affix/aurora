#include "affix-base/pch.h"
#include "weight.h"

using aurora::models::weight;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::params::param_sgd;
using aurora::maths::tensor;
using std::vector;
using std::initializer_list;
using std::uniform_real_distribution;

weight::~weight() {
	
}

weight::weight() {

}

void weight::param_recur(function<void(Param&)> a_func) {
	a_func(m_pmt);
}

model* weight::clone(function<Param(Param&)> a_func) {
	weight* result = new weight();
	result->m_pmt = a_func(m_pmt);
	return result;
}

void weight::fwd() {
	m_y.val() = m_x.val() * m_pmt->state();
}

void weight::bwd() {
	param_sgd* pmt_sgd = (param_sgd*)m_pmt.get();
	pmt_sgd->accum_grad(m_y_grad.val() * m_x.val());
	m_x_grad.val() = m_y_grad.val() * m_pmt->state();
}

void weight::signal(const tensor& a_y_des) {
	m_y_grad.val() = m_y.val() - a_y_des.val();
}

void weight::model_recur(function<void(model*)> a_func) {
	a_func(this);
}

void weight::compile() {

}