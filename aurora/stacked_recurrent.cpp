#include "affix-base/pch.h"
#include "stacked_recurrent.h"

using aurora::models::stacked_recurrent;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::params::param_sgd;
using aurora::maths::tensor;
using std::vector;
using std::initializer_list;
using std::uniform_real_distribution;

stacked_recurrent::~stacked_recurrent() {

}

stacked_recurrent::stacked_recurrent() {

}

stacked_recurrent::stacked_recurrent(vector<Recurrent> a_models) {
	m_models = a_models;
}

stacked_recurrent::stacked_recurrent(size_t a_height, Recurrent a_model_template) {
	for (int i = 0; i < a_height; i++)
		m_models.push_back((recurrent*)a_model_template->clone());
}

void stacked_recurrent::param_recur(const function<void(Param&)>& a_func) {
	for (int i = 0; i < m_models.size(); i++)
		m_models[i]->param_recur(a_func);
}

model* stacked_recurrent::clone(const function<Param(Param&)>& a_func) {
	stacked_recurrent* result = new stacked_recurrent();
	for (int i = 0; i < m_models.size(); i++)
		result->m_models.push_back((recurrent*)m_models[i]->clone(a_func));
	result->prep(m_prepared_size);
	result->unroll(m_unrolled_size);
	return result;
}

void stacked_recurrent::fwd() {
	for (int i = 0; i < m_models.size(); i++)
		m_models[i]->fwd();
}

void stacked_recurrent::bwd() {
	for (int i = m_models.size() - 1; i >= 0; i--)
		m_models[i]->bwd();
}

void stacked_recurrent::signal(const tensor& a_y_des) {
	m_models.back()->signal(a_y_des);
}

void stacked_recurrent::model_recur(const function<void(model*)>& a_func) {
	for (int i = 0; i < m_models.size(); i++)
		m_models[i]->model_recur(a_func);
}

void stacked_recurrent::compile() {
	m_x = tensor::new_1d(m_prepared_size);
	m_x_grad = tensor::new_1d(m_prepared_size);
	m_y = tensor::new_1d(m_prepared_size);
	m_y_grad = tensor::new_1d(m_prepared_size);

	tensor* l_x = &m_x;
	tensor* l_x_grad = &m_x_grad;

	for (int i = 0; i < m_models.size(); i++) {
		m_models[i]->compile();
		l_x->group_join(m_models[i]->m_x);
		l_x_grad->group_join(m_models[i]->m_x_grad);
		l_x = &m_models[i]->m_y;
		l_x_grad = &m_models[i]->m_y_grad;
	}

	m_y.group_join(*l_x);
	m_y_grad.group_join(*l_x_grad);
}

void stacked_recurrent::prep(size_t a_n) {
	m_prepared_size = a_n;
	for (int i = 0; i < m_models.size(); i++)
		m_models[i]->prep(a_n);
}

void stacked_recurrent::unroll(size_t a_n) {
	m_unrolled_size = a_n;
	for (int i = 0; i < m_models.size(); i++)
		m_models[i]->unroll(a_n);
}
