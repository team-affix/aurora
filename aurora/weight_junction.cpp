#include "affix-base/pch.h"
#include "weight_junction.h"

using aurora::models::weight_junction;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::params::param_sgd;
using aurora::maths::tensor;
using std::vector;
using std::initializer_list;
using std::uniform_real_distribution;

weight_junction::~weight_junction() {

}

weight_junction::weight_junction() {

}

weight_junction::weight_junction(size_t a_a, size_t a_b) {
	this->m_a = a_a;
	this->m_b = a_b;
	for (int i = 0; i < a_a; i++)
		m_weight_sets.push_back(new weight_set(a_b));
}

void weight_junction::param_recur(const function<void(Param&)>& a_func) {
	for (int i = 0; i < m_weight_sets.size(); i++)
		m_weight_sets[i]->param_recur(a_func);
}

model* weight_junction::clone(const function<Param(Param&)>& a_func) {
	weight_junction* result = new weight_junction();
	result->m_a = m_a;
	result->m_b = m_b;
	for (int i = 0; i < m_weight_sets.size(); i++)
		result->m_weight_sets.push_back((weight_set*)m_weight_sets[i]->clone(a_func));
	return result;
}

void weight_junction::fwd() {
	m_y.clear();
	for (int i = 0; i < m_weight_sets.size(); i++) {
		m_weight_sets[i]->fwd();
		m_y.add_1d(m_weight_sets[i]->m_y, m_y);
	}
}

void weight_junction::bwd() {
	for (int i = 0; i < m_weight_sets.size(); i++)
		m_weight_sets[i]->bwd();
}

void weight_junction::model_recur(const function<void(model*)>& a_func) {
	a_func(this);
	for (int i = 0; i < m_weight_sets.size(); i++)
		m_weight_sets[i]->model_recur(a_func);
}

void weight_junction::compile() {

	m_x = tensor::new_1d(m_a);
	m_x_grad = tensor::new_1d(m_a);
	m_y = tensor::new_1d(m_b);
	m_y_grad = tensor::new_1d(m_b);

	for (int i = 0; i < m_weight_sets.size(); i++) {
		m_weight_sets[i]->compile();
		m_weight_sets[i]->m_x.group_link(m_x[i]);
		m_weight_sets[i]->m_x_grad.group_link(m_x_grad[i]);
		m_weight_sets[i]->m_y_grad.group_link(m_y_grad);
	}
}
