#include "affix-base/pch.h"
#include "sigmoid.h"

using aurora::models::sigmoid;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::params::param_sgd;
using aurora::maths::tensor;
using std::vector;
using std::initializer_list;

sigmoid::~sigmoid() {

}

sigmoid::sigmoid() {

}

void sigmoid::param_recur(const function<void(Param&)>& a_func) {

}

model* sigmoid::clone(const function<Param(Param&)>& a_func) {
	return new sigmoid();
}

void sigmoid::fwd() {
	m_y.val() = (double)1 / ((double)1 + exp(-m_x.val()));
}

void sigmoid::bwd() {
	m_x_grad.val() = m_y_grad.val() * m_y.val() * (1 - m_y.val());
}

void sigmoid::model_recur(const function<void(model*)>& a_func) {
	a_func(this);
}

void sigmoid::compile() {

}
