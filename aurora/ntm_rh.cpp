#include "pch.h"
#include "ntm_rh.h"
#include "pseudo.h"

using aurora::models::ntm_rh;

ntm_rh::~ntm_rh() {

}

ntm_rh::ntm_rh() {
	units = 0;
	y_units = 0;
	s_units = 0;
	lr_units = 0;
	sm_units = 0;
}

ntm_rh::ntm_rh(vector<size_t> a_dims, size_t a_s_units, function<void(ptr<param>&)> a_func) {
	units = a_dims[0];
	s_units = a_s_units;
	lr_units = units + 2;
	sm_units = s_units + 1;
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
	a_dims.push_back(y_units);

	internal_model = pseudo::tnn(a_dims, pseudo::nlr(0.3), a_func);
	lr_layer = new layer(lr_units, pseudo::nlr(0.3), a_func);
	sm_layer = new layer(sm_units, pseudo::nsm(), a_func);
}

void ntm_rh::pmt_wise(function<void(ptr<param>&)> a_func) {
	internal_model->pmt_wise(a_func);
	lr_layer->pmt_wise(a_func);
	sm_layer->pmt_wise(a_func);
}

model* ntm_rh::clone() {
	ntm_rh* result = new ntm_rh();
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
	result->internal_model = internal_model->clone();
	result->lr_layer = (layer*)lr_layer->clone();
	result->sm_layer = (layer*)sm_layer->clone();
	return result;
}

model* ntm_rh::clone(function<void(ptr<param>&)> a_func) {
	ntm_rh* result = new ntm_rh();
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
	result->internal_model = internal_model->clone(a_func);
	result->lr_layer = (layer*)lr_layer->clone(a_func);
	result->sm_layer = (layer*)sm_layer->clone(a_func);
	return result;
}

void ntm_rh::fwd() {
	internal_model->fwd();
	lr_layer->fwd();
	sm_layer->fwd();
}

void ntm_rh::bwd() {
	lr_layer->bwd();
	sm_layer->bwd();
	internal_model->bwd();
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
	tensor lr_range = a_y_des.range(0, lr_units);
	tensor sm_range = a_y_des.range(lr_units, sm_units);
	lr_layer->signal(lr_range);
	sm_layer->signal(sm_range);
}

void ntm_rh::cycle(tensor& a_x, tensor& a_y_des) {
	x.pop(a_x);
	fwd();
	signal(a_y_des);
	bwd();
}

void ntm_rh::recur(function<void(model*)> a_func) {
	a_func(this);
	internal_model->recur(a_func);
	lr_layer->recur(a_func);
	sm_layer->recur(a_func);
}

void ntm_rh::compile() {
	
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

	k = lr_layer->y.range(0, units);
	k_grad = lr_layer->y_grad.range(0, units);
	beta = lr_layer->y.range(units, 1);
	beta_grad = lr_layer->y_grad.range(units, 1);
	gamma = lr_layer->y.range(units + 1, 1);
	gamma_grad = lr_layer->y_grad.range(units + 1, 1);
	g = sm_layer->y.range(0, 1);
	g_grad = sm_layer->y_grad.range(0, 1);
	s = sm_layer->y.range(1, s_units);
	s_grad = sm_layer->y_grad.range(1, s_units);

}