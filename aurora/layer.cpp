#include "affix-base/pch.h"
#include "layer.h"

using aurora::models::layer;
using std::initializer_list;
using aurora::params::Param;
using std::vector;
using std::function;
using aurora::models::model;
using aurora::maths::tensor;


layer::~layer() {

}

layer::layer() {

}

layer::layer(size_t a_height, Model a_model_template) {
	for (size_t i = 0; i < a_height; i++)
		m_models.push_back(a_model_template->clone([](Param& pmt) { return pmt->clone(); }));
}

layer::layer(initializer_list<Model> a_models) {
	for (initializer_list<Model>::iterator i = a_models.begin(); i != a_models.end(); i++)
		m_models.push_back(*i);
}

layer::layer(vector<Model> a_models) {
	m_models = a_models;
}

void layer::param_recur(const function<void(Param&)>& a_func) {
	for (int i = 0; i < m_models.size(); i++)
		m_models[i]->param_recur(a_func);
}

model* layer::clone(const function<Param(Param&)>& a_func) {
	layer* result = new layer();
	for (size_t i = 0; i < m_models.size(); i++)
		result->m_models.push_back(m_models[i]->clone(a_func));
	return result;
}

void layer::fwd() {
	for (size_t i = 0; i < m_models.size(); i++)
		m_models[i]->fwd();
}

void layer::bwd() {
	for (size_t i = 0; i < m_models.size(); i++)
		m_models[i]->bwd();
}

void layer::signal(const tensor& a_y_des) {
	for (size_t i = 0; i < m_models.size(); i++)
		m_models[i]->signal(a_y_des[i]);
}

void layer::model_recur(const function<void(model*)>& a_func) {
	a_func(this);
	for (size_t i = 0; i < m_models.size(); i++)
		m_models[i]->model_recur(a_func);
}

void layer::compile() {
	m_x.resize(m_models.size());
	m_y.resize(m_models.size());
	m_x_grad.resize(m_models.size());
	m_y_grad.resize(m_models.size());
	for (size_t i = 0; i < m_models.size(); i++) {
		m_models[i]->compile();
		m_models[i]->m_x.group_add(m_x[i]);
		m_models[i]->m_x_grad.group_add(m_x_grad[i]);
		m_models[i]->m_y.group_add(m_y[i]);
		m_models[i]->m_y_grad.group_add(m_y_grad[i]);
	}
}
