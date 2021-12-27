#include "affix-base/pch.h"
#include "sequential.h"

using aurora::models::sequential;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::params::param_sgd;
using aurora::maths::tensor;
using std::vector;
using std::initializer_list;

sequential::~sequential() {

}

sequential::sequential() {

}

sequential::sequential(initializer_list<Model> a_models) {
	for (initializer_list<Model>::iterator i = a_models.begin(); i != a_models.end(); i++)
		m_models.push_back(*i);
}

sequential::sequential(vector<Model> a_models) {
	m_models = a_models;
}

void sequential::param_recur(const function<void(Param&)>& a_func) {
	for (int i = 0; i < m_models.size(); i++)
		m_models[i]->param_recur(a_func);
}

model* sequential::clone(const function<Param(Param&)>& a_func) {
	sequential* result = new sequential();
	result->m_models.resize(m_models.size());
	for (int i = 0; i < m_models.size(); i++)
		result->m_models[i] = m_models[i]->clone(a_func);
	return result;
}

void sequential::fwd() {
	for (int i = 0; i < m_models.size(); i++)
		m_models[i]->fwd();
}

void sequential::bwd() {
	for (int i = m_models.size() - 1; i >= 0; i--)
		m_models[i]->bwd();
}

void sequential::signal(const tensor& a_y_des) {
	m_models.back()->signal(a_y_des);
}

void sequential::model_recur(const function<void(model*)>& a_func) {
	a_func(this);
	for (int i = 0; i < m_models.size(); i++)
		m_models[i]->model_recur(a_func);
}

void sequential::compile() {
	tensor* state = &m_x;
	tensor* gradient = &m_x_grad;
	for (int i = 0; i < m_models.size(); i++) {
		m_models[i]->compile();
		state->group_join_all_ranks(m_models[i]->m_x);
		gradient->group_join_all_ranks(m_models[i]->m_x_grad);
		state = &m_models[i]->m_y;
		gradient = &m_models[i]->m_y_grad;
	}
	state->group_add_all_ranks(m_y);
	gradient->group_add_all_ranks(m_y_grad);
}
