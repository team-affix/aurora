#include "pch.h"
#include "ntm_rh.h"
#include "pseudo.h"
#include "normalize.h"

using aurora::models::ntm_rh;

ntm_rh::~ntm_rh() {

}

ntm_rh::ntm_rh() {

}

ntm_rh::ntm_rh(size_t a_units, vector<size_t> a_head_h_dims, size_t a_shift_units, function<void(ptr<param>&)> a_func) {
	units = a_units;
	shift_units = a_shift_units;

	y_units += units; // KEY
	y_units += 1; // BETA
	y_units += 1; // G
	y_units += shift_units; // S
	y_units += 1; // GAMMA

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

	vector<ptr<model>> beta_neurons;
	vector<ptr<model>> g_neurons;
	vector<ptr<model>> s_neurons;
	vector<ptr<model>> gamma_neurons;

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

	key_model = pseudo::tnn(key_dims, pseudo::nlr(0.3), a_func);
	beta_model = pseudo::tnn(beta_dims, beta_neurons, a_func);
	g_model = pseudo::tnn(g_dims, g_neurons, a_func);
	s_model = new sequential{ pseudo::tnn(s_dims, s_neurons, a_func), new normalize(a_shift_units) };
	gamma_model = pseudo::tnn(gamma_dims, gamma_neurons, a_func);
}

void ntm_rh::pmt_wise(function<void(ptr<param>&)> a_func) {
	key_model->pmt_wise(a_func);
	beta_model->pmt_wise(a_func);
	g_model->pmt_wise(a_func);
	s_model->pmt_wise(a_func);
	gamma_model->pmt_wise(a_func);
}

model* ntm_rh::clone() {
	ntm_rh* result = new ntm_rh();
	result->units = units;
	result->shift_units = shift_units;
	result->y_units = y_units;
	result->key_model = key_model->clone();
	result->beta_model = beta_model->clone();
	result->g_model = g_model->clone();
	result->s_model = s_model->clone();
	result->gamma_model = gamma_model->clone();
	return result;
}

model* ntm_rh::clone(function<void(ptr<param>&)> a_func) {
	ntm_rh* result = new ntm_rh();
	result->units = units;
	result->shift_units = shift_units;
	result->y_units = y_units;
	result->key_model = key_model->clone(a_func);
	result->beta_model = beta_model->clone(a_func);
	result->g_model = g_model->clone(a_func);
	result->s_model = s_model->clone(a_func);
	result->gamma_model = gamma_model->clone(a_func);
	return result;
}

void ntm_rh::fwd() {
	key_model->fwd();
	beta_model->fwd();
	g_model->fwd();
	s_model->fwd();
	gamma_model->fwd();
}

void ntm_rh::bwd() {
	key_model->bwd();
	beta_model->bwd();
	g_model->bwd();
	s_model->bwd();
	gamma_model->bwd();
	x_grad.clear();
	x_grad.add_1d(key_model->x_grad, x_grad);
	x_grad.add_1d(beta_model->x_grad, x_grad);
	x_grad.add_1d(g_model->x_grad, x_grad);
	x_grad.add_1d(s_model->x_grad, x_grad);
	x_grad.add_1d(gamma_model->x_grad, x_grad);
}

tensor& ntm_rh::fwd(tensor& a_x) {
	x.pop(a_x);
	fwd();
	return y;
}

tensor& ntm_rh::bwd(tensor& a_y_grad) {
	y_grad.pop(a_y_grad);
	bwd();
	return x_grad;
}

void ntm_rh::signal(tensor& a_y_des) {
	y.sub_1d(a_y_des, y_grad);
}

void ntm_rh::cycle(tensor& a_x, tensor& a_y_des) {
	x.pop(a_x);
	fwd();
	signal(a_y_des);
	bwd();
}

void ntm_rh::recur(function<void(model*)> a_func) {
	a_func(this);
	key_model->recur(a_func);
	beta_model->recur(a_func);
	g_model->recur(a_func);
	s_model->recur(a_func);
	gamma_model->recur(a_func);
}

void ntm_rh::compile() {
	x = tensor::new_1d(units);
	x_grad = tensor::new_1d(units);
	y = tensor::new_1d(y_units);
	y_grad = tensor::new_1d(y_units);
	key = tensor::new_1d(units);
	key_grad = tensor::new_1d(units);
	beta = tensor::new_1d(1);
	beta_grad = tensor::new_1d(1);
	g = tensor::new_1d(1);
	g_grad = tensor::new_1d(1);
	s = tensor::new_1d(shift_units);
	s_grad = tensor::new_1d(shift_units);
	gamma = tensor::new_1d(1);
	gamma_grad = tensor::new_1d(1);
	
	key_model->compile();
	beta_model->compile();
	g_model->compile();
	s_model->compile();
	gamma_model->compile();

	key_model->x.group_join(x);
	beta_model->x.group_join(x);
	g_model->x.group_join(x);
	s_model->x.group_join(x);
	gamma_model->x.group_join(x);

	key.group_join_all_ranks(key_model->y);
	key_grad.group_join_all_ranks(key_model->y_grad);
	beta.group_join_all_ranks(beta_model->y);
	beta_grad.group_join_all_ranks(beta_model->y_grad);
	g.group_join_all_ranks(g_model->y);
	g_grad.group_join_all_ranks(g_model->y_grad);
	s.group_join_all_ranks(s_model->y);
	s_grad.group_join_all_ranks(s_model->y_grad);
	gamma.group_join_all_ranks(gamma_model->y);
	gamma_grad.group_join_all_ranks(gamma_model->y_grad);

	y = key.cat(beta).cat(g).cat(s).cat(gamma);
	y_grad = key_grad.cat(beta_grad).cat(g_grad).cat(s_grad).cat(gamma_grad);
}