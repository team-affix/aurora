#include "affix-base/pch.h"
#include "weight_set.h"

using aurora::models::weight_set;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::params::param_sgd;
using aurora::maths::tensor;
using std::vector;
using std::initializer_list;
using std::uniform_real_distribution;

weight_set::~weight_set() {

}

weight_set::weight_set() {

}

weight_set::weight_set(size_t a_a) {
	this->m_a = a_a;
	for (int i = 0; i < a_a; i++)
		m_weights.push_back(new weight());
}

void weight_set::param_recur(const function<void(Param&)>& a_func) {
	for (int i = 0; i < m_weights.size(); i++)
		m_weights[i]->param_recur(a_func);
}

model* weight_set::clone(const function<Param(Param&)>& a_func) {
	weight_set* result = new weight_set();
	result->m_a = m_a;
	for (int i = 0; i < m_a; i++)
		result->m_weights.push_back((weight*)m_weights[i]->clone(a_func));
	return result;
}

void weight_set::fwd() {
	for (int i = 0; i < m_weights.size(); i++)
		m_weights[i]->fwd();
}

void weight_set::bwd() {
	m_x_grad.val() = 0;
	for (int i = 0; i < m_weights.size(); i++) {
		m_weights[i]->bwd();
		m_x_grad.val() += m_weights[i]->m_x_grad.val();
	}
}

void weight_set::model_recur(const function<void(model*)>& a_func) {
	a_func(this);
	for (int i = 0; i < m_weights.size(); i++)
		m_weights[i]->model_recur(a_func);
}

void weight_set::compile() {
	m_y.resize(m_a);
	m_y_grad.resize(m_a);
	for (int i = 0; i < m_a; i++) {
		m_weights[i]->compile();
		m_weights[i]->m_x.link(m_x);
		m_weights[i]->m_y.link(m_y[i]);
		m_weights[i]->m_y_grad.link(m_y_grad[i]);
	}
}
