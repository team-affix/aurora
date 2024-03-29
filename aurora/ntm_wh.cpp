#include "affix-base/pch.h"
#include "ntm_wh.h"
#include "pseudo_tnn.h"
#include "neuron.h"

using aurora::models::ntm_wh;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::params::param_sgd;
using aurora::maths::tensor;
using std::vector;

ntm_wh::~ntm_wh() {

}

ntm_wh::ntm_wh() {

}

ntm_wh::ntm_wh(size_t a_units, vector<size_t> a_head_h_dims, size_t a_shift_units) {
	m_units = a_units;

	vector<size_t> dims = a_head_h_dims;
	dims.insert(dims.begin(), a_units);
	dims.push_back(a_units);

	vector<Model> e_neurons;

	for (int i = 0; i < dims.size() - 1; i++) {
		e_neurons.push_back(pseudo::nth());
	}
	e_neurons.push_back(pseudo::nsm());

	m_internal_rh = new ntm_rh(a_units, a_head_h_dims, a_shift_units);
	m_a_model = pseudo::tnn(dims, pseudo::nlr(0.3));
	m_e_model = pseudo::tnn(dims, e_neurons);
}

void ntm_wh::param_recur(const function<void(Param&)>& a_func) {
	m_internal_rh->param_recur(a_func);
	m_a_model->param_recur(a_func);
	m_e_model->param_recur(a_func);
}

model* ntm_wh::clone(const function<Param(Param&)>& a_func) {
	ntm_wh* result = new ntm_wh();
	result->m_units = m_units;
	result->m_internal_rh = (ntm_rh*)m_internal_rh->clone(a_func);
	result->m_a_model = m_a_model->clone(a_func);
	result->m_e_model = m_e_model->clone(a_func);
	return result;
}

void ntm_wh::fwd() {
	m_internal_rh->fwd();
	m_a_model->fwd();
	m_e_model->fwd();
}

void ntm_wh::bwd() {
	m_internal_rh->bwd();
	m_a_model->bwd();
	m_e_model->bwd();
	m_x_grad.clear();
	m_x_grad.add_1d(m_internal_rh->m_x_grad, m_x_grad);
	m_x_grad.add_1d(m_a_model->m_x_grad, m_x_grad);
	m_x_grad.add_1d(m_e_model->m_x_grad, m_x_grad);
}

void ntm_wh::model_recur(const function<void(model*)>& a_func) {
	a_func(this);
	m_internal_rh->model_recur(a_func);
	m_a_model->model_recur(a_func);
	m_e_model->model_recur(a_func);
}

void ntm_wh::compile() {
	m_x = tensor::new_1d(m_units);
	m_x_grad = tensor::new_1d(m_units);
	m_y = tensor::new_1d(m_internal_rh->m_y_units);
	m_y_grad = tensor::new_1d(m_internal_rh->m_y_units);
	m_a = tensor::new_1d(m_units);
	m_a_grad = tensor::new_1d(m_units);
	m_e = tensor::new_1d(m_units);
	m_e_grad = tensor::new_1d(m_units);

	m_internal_rh->compile();
	m_a_model->compile();
	m_e_model->compile();

	m_x.group_link(m_internal_rh->m_x);
	m_x.group_link(m_a_model->m_x);
	m_x.group_link(m_e_model->m_x);

	m_a.group_link(m_a_model->m_y);
	m_a_grad.group_link(m_a_model->m_y_grad);
	m_e.group_link(m_e_model->m_y);
	m_e_grad.group_link(m_e_model->m_y_grad);

	m_internal_rh->m_y.group_link(m_y);
	m_internal_rh->m_y_grad.group_link(m_y_grad);
}
