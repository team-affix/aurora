#include "affix-base/pch.h"
#include "ntm_rh.h"
#include "pseudo_tnn.h"
#include "neuron.h"
#include "normalize.h"

using aurora::models::ntm_rh;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::params::param_sgd;
using aurora::maths::tensor;
using std::vector;

ntm_rh::~ntm_rh() {

}

ntm_rh::ntm_rh() {

}

ntm_rh::ntm_rh(size_t a_units, vector<size_t> a_head_h_dims, size_t a_shift_units) {
	m_units = a_units;
	m_shift_units = a_shift_units;

	m_y_units += m_units; // KEY
	m_y_units += 1; // BETA
	m_y_units += 1; // G
	m_y_units += m_shift_units; // S
	m_y_units += 1; // GAMMA

	vector<size_t> dims = a_head_h_dims;
	dims.insert(dims.begin(), a_units);

	vector<size_t> key_dims = dims;
	key_dims.push_back(a_units);
	vector<size_t> beta_dims = dims;
	beta_dims.push_back(1);
	vector<size_t> g_dims = dims;
	g_dims.push_back(1);
	vector<size_t> s_dims = dims;
	s_dims.push_back(a_shift_units);
	vector<size_t> gamma_dims = dims;
	gamma_dims.push_back(1);

	vector<Model> beta_neurons;
	vector<Model> g_neurons;
	vector<Model> s_neurons;
	vector<Model> gamma_neurons;

	// DO NOT LOOP THROUGH OUTPUT LAYER
	for (int i = 0; i < dims.size(); i++) {
		beta_neurons.push_back(pseudo::nth());
		g_neurons.push_back(pseudo::nth());
		s_neurons.push_back(pseudo::nth());
		gamma_neurons.push_back(pseudo::nth());
	}
	beta_neurons.push_back(pseudo::nlrexu(1));
	g_neurons.push_back(pseudo::nsm());
	s_neurons.push_back(pseudo::nsm());
	gamma_neurons.push_back(pseudo::nlrexu(2));

	m_key_model = pseudo::tnn(key_dims, pseudo::nlr(0.3));
	m_beta_model = pseudo::tnn(beta_dims, beta_neurons);
	m_g_model = pseudo::tnn(g_dims, g_neurons);
	m_s_model = new sequential{ pseudo::tnn(s_dims, s_neurons), new normalize(a_shift_units) };
	m_gamma_model = pseudo::tnn(gamma_dims, gamma_neurons);
}

void ntm_rh::param_recur(const function<void(Param&)>& a_func) {
	m_key_model->param_recur(a_func);
	m_beta_model->param_recur(a_func);
	m_g_model->param_recur(a_func);
	m_s_model->param_recur(a_func);
	m_gamma_model->param_recur(a_func);
}

model* ntm_rh::clone(const function<Param(Param&)>& a_func) {
	ntm_rh* result = new ntm_rh();
	result->m_units = m_units;
	result->m_shift_units = m_shift_units;
	result->m_y_units = m_y_units;
	result->m_key_model = m_key_model->clone(a_func);
	result->m_beta_model = m_beta_model->clone(a_func);
	result->m_g_model = m_g_model->clone(a_func);
	result->m_s_model = m_s_model->clone(a_func);
	result->m_gamma_model = m_gamma_model->clone(a_func);
	return result;
}

void ntm_rh::fwd() {
	m_key_model->fwd();
	m_beta_model->fwd();
	m_g_model->fwd();
	m_s_model->fwd();
	m_gamma_model->fwd();
}

void ntm_rh::bwd() {
	m_key_model->bwd();
	m_beta_model->bwd();
	m_g_model->bwd();
	m_s_model->bwd();
	m_gamma_model->bwd();
	m_x_grad.clear();
	m_x_grad.add_1d(m_key_model->m_x_grad, m_x_grad);
	m_x_grad.add_1d(m_beta_model->m_x_grad, m_x_grad);
	m_x_grad.add_1d(m_g_model->m_x_grad, m_x_grad);
	m_x_grad.add_1d(m_s_model->m_x_grad, m_x_grad);
	m_x_grad.add_1d(m_gamma_model->m_x_grad, m_x_grad);
}

void ntm_rh::signal(const tensor& a_y_des) {
	m_y.sub_1d(a_y_des, m_y_grad);
}

void ntm_rh::model_recur(const function<void(model*)>& a_func) {
	a_func(this);
	m_key_model->model_recur(a_func);
	m_beta_model->model_recur(a_func);
	m_g_model->model_recur(a_func);
	m_s_model->model_recur(a_func);
	m_gamma_model->model_recur(a_func);
}

void ntm_rh::compile() {
	m_x = tensor::new_1d(m_units);
	m_x_grad = tensor::new_1d(m_units);
	m_y = tensor::new_1d(m_y_units);
	m_y_grad = tensor::new_1d(m_y_units);
	m_key = tensor::new_1d(m_units);
	m_key_grad = tensor::new_1d(m_units);
	m_beta = tensor::new_1d(1);
	m_beta_grad = tensor::new_1d(1);
	m_g = tensor::new_1d(1);
	m_g_grad = tensor::new_1d(1);
	m_s = tensor::new_1d(m_shift_units);
	m_s_grad = tensor::new_1d(m_shift_units);
	m_gamma = tensor::new_1d(1);
	m_gamma_grad = tensor::new_1d(1);
	
	m_key_model->compile();
	m_beta_model->compile();
	m_g_model->compile();
	m_s_model->compile();
	m_gamma_model->compile();

	m_key_model->m_x.group_join(m_x);
	m_beta_model->m_x.group_join(m_x);
	m_g_model->m_x.group_join(m_x);
	m_s_model->m_x.group_join(m_x);
	m_gamma_model->m_x.group_join(m_x);

	m_key.group_join_all_ranks(m_key_model->m_y);
	m_key_grad.group_join_all_ranks(m_key_model->m_y_grad);
	m_beta.group_join_all_ranks(m_beta_model->m_y);
	m_beta_grad.group_join_all_ranks(m_beta_model->m_y_grad);
	m_g.group_join_all_ranks(m_g_model->m_y);
	m_g_grad.group_join_all_ranks(m_g_model->m_y_grad);
	m_s.group_join_all_ranks(m_s_model->m_y);
	m_s_grad.group_join_all_ranks(m_s_model->m_y_grad);
	m_gamma.group_join_all_ranks(m_gamma_model->m_y);
	m_gamma_grad.group_join_all_ranks(m_gamma_model->m_y_grad);

	m_y = m_key.cat(m_beta).cat(m_g).cat(m_s).cat(m_gamma);
	m_y_grad = m_key_grad.cat(m_beta_grad).cat(m_g_grad).cat(m_s_grad).cat(m_gamma_grad);
}
