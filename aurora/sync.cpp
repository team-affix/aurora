#include "affix-base/pch.h"
#include "sync.h"

using aurora::models::sync;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::params::param_sgd;
using aurora::maths::tensor;
using std::vector;
using std::initializer_list;
using std::uniform_real_distribution;

sync::~sync() {

}

sync::sync(Model a_model_template) {
	this->m_model_template = a_model_template;
}

void sync::param_recur(const function<void(Param&)>& a_func) {
	m_model_template->param_recur(a_func);
}

model* sync::clone(const function<Param(Param&)>& a_func) {
	sync* result = new sync(m_model_template->clone(a_func));
	result->prep(m_prepared.size());
	result->unroll(m_unrolled.size());
	return result;
}

void sync::fwd() {
	for (int i = 0; i < m_unrolled.size(); i++)
		m_unrolled[i]->fwd();
}

void sync::bwd() {
	for (int i = 0; i < m_unrolled.size(); i++)
		m_unrolled[i]->bwd();
}

void sync::signal(const tensor& a_y_des) {
	for (int i = 0; i < m_unrolled.size(); i++)
		m_unrolled[i]->signal(a_y_des[i]);
}

void sync::model_recur(const function<void(model*)>& a_func) {
	a_func(this);
	m_model_template->model_recur(a_func);
}

void sync::compile() {
	m_x.resize(m_prepared.size());
	m_y.resize(m_prepared.size());
	m_x_grad.resize(m_prepared.size());
	m_y_grad.resize(m_prepared.size());
	for (int i = 0; i < m_prepared.size(); i++) {
		m_prepared[i]->compile();
		m_x[i].group_join_all_ranks(m_prepared[i]->m_x);
		m_y[i].group_join_all_ranks(m_prepared[i]->m_y);
		m_x_grad[i].group_join_all_ranks(m_prepared[i]->m_x_grad);
		m_y_grad[i].group_join_all_ranks(m_prepared[i]->m_y_grad);
	}
}

void sync::prep(size_t a_n) {
	m_prepared.clear();
	m_prepared.resize(a_n);
	for (int i = 0; i < a_n; i++)
		m_prepared.at(i) = m_model_template->clone();
}

void sync::unroll(size_t a_n) {
	m_unrolled.clear();
	m_unrolled.resize(a_n);
	for (int i = 0; i < a_n; i++)
		m_unrolled.at(i) = m_prepared.at(i);
}
