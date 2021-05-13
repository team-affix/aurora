#include "pch.h"
#include "ntm_wh.h"
#include "weight.h"
#include "pseudo.h"

using aurora::models::ntm_wh;
using aurora::models::weight;

ntm_wh::~ntm_wh() {

}

ntm_wh::ntm_wh() {

}

ntm_wh::ntm_wh(vector<size_t> a_dims, size_t a_s_units, function<void(ptr<param>&)> a_func) {
	units = a_dims[0];
	s_units = a_s_units;
	lr_units = units + units + 2;
	sm_units = s_units + units + 1;
	y_units = lr_units + sm_units;

	vector<size_t> lr_dims = a_dims;
	vector<size_t> sm_dims = a_dims;

	lr_dims.push_back(lr_units);
	sm_dims.push_back(sm_units);

	vector<ptr<model>> sm_neurons(a_dims.size() + 1);
	for (int i = 0; i < sm_neurons.size() - 1; i++)
		sm_neurons[i] = pseudo::nth();
	sm_neurons.back() = pseudo::nsm();

	lr_model = pseudo::tnn(lr_dims, pseudo::nlr(0.3), a_func);
	sm_model = pseudo::tnn(sm_dims, sm_neurons, a_func);

}

model* ntm_wh::clone() {
	ntm_wh* result = new ntm_wh();
	result->units = units;
	result->s_units = s_units;
	result->lr_units = lr_units;
	result->sm_units = sm_units;
	result->k = k.clone();
	result->k_grad = k_grad.clone();
	result->beta = beta.clone();
	result->beta_grad = beta_grad.clone();
	result->g = g.clone();
	result->g_grad = g_grad.clone();
	result->s = s.clone();
	result->s_grad = s_grad.clone();
	result->gamma = s.clone();
	result->gamma_grad = s_grad.clone();
	result->e = e.clone();
	result->e_grad = e_grad.clone();
	result->a = a.clone();
	result->a_grad = a_grad.clone();
	result->lr_model = (layer*)lr_model->clone();
	result->sm_model = (layer*)sm_model->clone();
	return result;
}

model* ntm_wh::clone(function<void(ptr<param>&)> a_func) {
	ntm_wh* result = new ntm_wh();
	result->units = units;
	result->s_units = s_units;
	result->lr_units = lr_units;
	result->sm_units = sm_units;
	result->k = k.clone();
	result->k_grad = k_grad.clone();
	result->beta = beta.clone();
	result->beta_grad = beta_grad.clone();
	result->g = g.clone();
	result->g_grad = g_grad.clone();
	result->s = s.clone();
	result->s_grad = s_grad.clone();
	result->gamma = s.clone();
	result->gamma_grad = s_grad.clone();
	result->e = e.clone();
	result->e_grad = e_grad.clone();
	result->a = a.clone();
	result->a_grad = a_grad.clone();
	result->lr_model = (layer*)lr_model->clone(a_func);
	result->sm_model = (layer*)sm_model->clone(a_func);
	return result;
}

void ntm_wh::compile() {

	x = tensor::new_1d(units);
	x_grad = tensor::new_1d(units);
	y = tensor::new_1d(y_units);
	y_grad = tensor::new_1d(y_units);

	k = tensor::new_1d(units);
	k_grad = tensor::new_1d(units);
	beta = tensor::new_1d(1);
	beta_grad = tensor::new_1d(1);
	g = tensor::new_1d(1);
	g_grad = tensor::new_1d(1);
	s = tensor::new_1d(s_units);
	s_grad = tensor::new_1d(s_units);
	gamma = tensor::new_1d(1);
	gamma_grad = tensor::new_1d(1);
	e = tensor::new_1d(units);
	e_grad = tensor::new_1d(units);
	a = tensor::new_1d(units);
	a_grad = tensor::new_1d(units);

	lr_model->compile();
	sm_model->compile();

	tensor y_range = lr_model->y.concat(sm_model->y);
	tensor y_grad_range = lr_model->y_grad.concat(sm_model->y_grad);

	x.group_join(lr_model->x);
	x.group_join(sm_model->x);
	y.group_join_all_ranks(y_range);
	y_grad.group_join_all_ranks(y_grad_range);

	size_t lr_y = 0;

	k = lr_model->y.range(lr_y, units);
	k_grad = lr_model->y_grad.range(lr_y, units);
	lr_y += units;

	beta = lr_model->y.range(lr_y, 1);
	beta_grad = lr_model->y_grad.range(lr_y, 1);
	lr_y += 1;

	gamma = lr_model->y.range(lr_y, 1);
	gamma_grad = lr_model->y_grad.range(lr_y, 1);
	lr_y += 1;

	a = lr_model->y.range(lr_y, units);
	a_grad = lr_model->y_grad.range(lr_y, units);
	lr_y += units;

	size_t sm_y = 0;

	g = sm_model->y.range(sm_y, 1);
	g_grad = sm_model->y_grad.range(sm_y, 1);
	sm_y += 1;

	s = sm_model->y.range(sm_y, s_units);
	s_grad = sm_model->y_grad.range(sm_y, s_units);
	sm_y += s_units;

	e = sm_model->y.range(sm_y, units);
	e_grad = sm_model->y_grad.range(sm_y, units);
	sm_y += units;

}