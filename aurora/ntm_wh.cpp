#include "pch.h"
#include "ntm_wh.h"
#include "pseudo.h"

using aurora::models::ntm_wh;

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

	a_dims.push_back(y_units);

	internal_model = pseudo::tnn(a_dims, pseudo::nth(), a_func);
	lr_layer = new layer(lr_units, pseudo::nlr(0.3), a_func);
	sm_layer = new layer(sm_units, pseudo::nsm(), a_func);

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
	result->internal_model = internal_model->clone();
	result->lr_layer = (layer*)lr_layer->clone();
	result->sm_layer = (layer*)sm_layer->clone();
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
	result->internal_model = internal_model->clone(a_func);
	result->lr_layer = (layer*)lr_layer->clone(a_func);
	result->sm_layer = (layer*)sm_layer->clone(a_func);
	return result;
}

void ntm_wh::compile() {

	x = tensor::new_1d(units);
	x_grad = tensor::new_1d(units);
	y = tensor::new_1d(y_units);
	y_grad = tensor::new_1d(y_units);

	internal_model->compile();
	lr_layer->compile();
	sm_layer->compile();

	tensor y_range = lr_layer->y.concat(sm_layer->y);
	tensor y_grad_range = lr_layer->y_grad.concat(sm_layer->y_grad);

	x.group_join(internal_model->x);
	x_grad.group_join(internal_model->x_grad);
	y.group_join_all_ranks(y_range);
	y_grad.group_join_all_ranks(y_grad_range);

	tensor y_lr_range = internal_model->y.range(0, lr_units);
	tensor y_grad_lr_range = internal_model->y_grad.range(0, lr_units);
	tensor y_sm_range = internal_model->y.range(lr_units, sm_units);
	tensor y_grad_sm_range = internal_model->y_grad.range(0, lr_units);

	lr_layer->x.group_join_all_ranks(y_lr_range);
	lr_layer->x_grad.group_join_all_ranks(y_grad_lr_range);
	sm_layer->x.group_join_all_ranks(y_sm_range);
	sm_layer->x_grad.group_join_all_ranks(y_grad_sm_range);

	size_t lr_y = 0;

	k = lr_layer->y.range(lr_y, units);
	k_grad = lr_layer->y_grad.range(lr_y, units);
	lr_y += units;

	beta = lr_layer->y.range(lr_y, 1);
	beta_grad = lr_layer->y_grad.range(lr_y, 1);
	lr_y += 1;

	gamma = lr_layer->y.range(lr_y, 1);
	gamma_grad = lr_layer->y_grad.range(lr_y, 1);
	lr_y += 1;

	a = lr_layer->y.range(lr_y, units);
	a_grad = lr_layer->y_grad.range(lr_y, units);
	lr_y += units;

	size_t sm_y = 0;

	g = sm_layer->y.range(sm_y, 1);
	g_grad = sm_layer->y_grad.range(sm_y, 1);
	sm_y += 1;

	s = sm_layer->y.range(sm_y, s_units);
	s_grad = sm_layer->y_grad.range(sm_y, s_units);
	sm_y += s_units;

	e = sm_layer->y.range(sm_y, units);
	e_grad = sm_layer->y_grad.range(sm_y, units);
	sm_y += units;

}