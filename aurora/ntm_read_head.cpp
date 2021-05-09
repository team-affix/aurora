#include "pch.h"
#include "ntm_read_head.h"
#include "pseudo.h"

using aurora::models::ntm_read_head;

ntm_read_head::~ntm_read_head() {

}

ntm_read_head::ntm_read_head() {
	units = 0;
	shift_units = 0;
}

ntm_read_head::ntm_read_head(vector<size_t> a_dims, size_t a_shift_units, function<void(ptr<param>&)> a_func) {
	units = a_dims[0];
	shift_units = a_shift_units;
	lr_units = units + 2;
	sm_units = shift_units + 1;

	size_t y_units = lr_units + sm_units;

	k = tensor::new_1d(units);
	k_grad = tensor::new_1d(units);
	beta = tensor::new_1d(1);
	beta_grad = tensor::new_1d(1);
	g = tensor::new_1d(1);
	g_grad = tensor::new_1d(1);
	s = tensor::new_1d(shift_units);
	s_grad = tensor::new_1d(shift_units);
	gamma = tensor::new_1d(1);
	gamma_grad = tensor::new_1d(1);
	a_dims.push_back(y_units);

	internal_model = pseudo::tnn(a_dims, pseudo::nlr(0.3), a_func);
	lr_layer = new layer(lr_units, pseudo::nlr(0.3), a_func);
	sm_layer = new layer(sm_units, pseudo::nsm(), a_func);
}

void ntm_read_head::pmt_wise(function<void(ptr<param>&)> a_func) {
	internal_model->pmt_wise(a_func);
	lr_layer->pmt_wise(a_func);
	sm_layer->pmt_wise(a_func);
}

model* ntm_read_head::clone() {
	ntm_read_head* result = new ntm_read_head();
	result->units = units;
	result->shift_units = shift_units;
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

model* ntm_read_head::clone(function<void(ptr<param>&)> a_func) {
	ntm_read_head* result = new ntm_read_head();
	result->units = units;
	result->shift_units = shift_units;
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

void ntm_read_head::fwd() {
	internal_model->fwd();
	lr_layer->fwd();
	sm_layer->fwd();
}

void ntm_read_head::bwd() {
	lr_layer->bwd();
	sm_layer->bwd();
	internal_model->bwd();
}

tensor& ntm_read_head::fwd(tensor& a_x) {
	x.pop(a_x);
	fwd();
	return y;
}

tensor& ntm_read_head::bwd(tensor& a_y_grad) {
	y_grad.pop(a_y_grad);
	bwd();
	return x_grad;
}

void ntm_read_head::signal(tensor& a_y_des) {
	tensor lr_range = a_y_des.range(0, lr_units);
	tensor sm_range = a_y_des.range(lr_units, sm_units);
	lr_layer->signal(lr_range);
	sm_layer->signal(sm_range);
}

void ntm_read_head::cycle(tensor& a_x, tensor& a_y_des) {
	x.pop(a_x);
	fwd();
	signal(a_y_des);
	bwd();
}

void ntm_read_head::recur(function<void(model*)> a_func) {
	a_func(this);
	internal_model->recur(a_func);
	lr_layer->recur(a_func);
	sm_layer->recur(a_func);
}

void ntm_read_head::compile() {
	internal_model->compile();
	lr_layer->compile();
	sm_layer->compile();
	tensor y_range = lr_layer->y.concat(sm_layer->y);
	tensor y_grad_range = lr_layer->y_grad.concat(sm_layer->y_grad);
	x.group_join(internal_model->x);
	x_grad.group_join(internal_model->x_grad);
	y.group_join_all_ranks(y_range);
	y_grad.group_join_all_ranks(y_grad_range);	

}