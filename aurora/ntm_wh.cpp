#include "pch.h"
#include "ntm_wh.h"
#include "pseudo.h"

using aurora::models::ntm_wh;

ntm_wh::~ntm_wh() {

}

ntm_wh::ntm_wh() {

}

ntm_wh::ntm_wh(size_t a_units, vector<size_t> a_head_h_dims, size_t a_shift_units, function<void(ptr<param>&)> a_func) {
	units = a_units;

	vector<size_t> dims = a_head_h_dims;
	dims.insert(dims.begin(), a_units);
	dims.push_back(a_units);

	vector<ptr<model>> e_neurons;

	for (int i = 0; i < dims.size() - 1; i++) {
		e_neurons.push_back(pseudo::nth());
	}
	e_neurons.push_back(pseudo::nsm());

	internal_rh = new ntm_rh(a_units, a_head_h_dims, a_shift_units, a_func);
	a_model = pseudo::tnn(dims, pseudo::nlr(0.3), a_func);
	e_model = pseudo::tnn(dims, e_neurons, a_func);
}

void ntm_wh::pmt_wise(function<void(ptr<param>&)> a_func) {
	internal_rh->pmt_wise(a_func);
	a_model->pmt_wise(a_func);
	e_model->pmt_wise(a_func);
}

model* ntm_wh::clone() {
	ntm_wh* result = new ntm_wh();
	result->units = units;
	result->internal_rh = (ntm_rh*)internal_rh->clone();
	result->a_model = a_model->clone();
	result->e_model = e_model->clone();
	return result;
}

model* ntm_wh::clone(function<void(ptr<param>&)> a_func) {
	ntm_wh* result = new ntm_wh();
	result->units = units;
	result->internal_rh = (ntm_rh*)internal_rh->clone(a_func);
	result->a_model = a_model->clone(a_func);
	result->e_model = e_model->clone(a_func);
	return result;
}

void ntm_wh::fwd() {
	internal_rh->fwd();
	a_model->fwd();
	e_model->fwd();
}

void ntm_wh::bwd() {
	internal_rh->bwd();
	a_model->bwd();
	e_model->bwd();
	x_grad.clear();
	x_grad.add_1d(internal_rh->x_grad, x_grad);
	x_grad.add_1d(a_model->x_grad, x_grad);
	x_grad.add_1d(e_model->x_grad, x_grad);
}

tensor& ntm_wh::fwd(tensor& a_x) {
	x.pop(a_x);
	fwd();
	return y;
}

tensor& ntm_wh::bwd(tensor& a_y_grad) {
	y_grad.pop(a_y_grad);
	bwd();
	return x_grad;
}

void ntm_wh::signal(tensor& a_y_des) {
	y.sub_1d(a_y_des, y_grad);
}

void ntm_wh::cycle(tensor& a_x, tensor& a_y_des) {
	x.pop(a_x);
	fwd();
	signal(a_y_des);
	bwd();
}

void ntm_wh::recur(function<void(model*)> a_func) {
	a_func(this);
	internal_rh->recur(a_func);
	a_model->recur(a_func);
	e_model->recur(a_func);
}

void ntm_wh::compile() {
	x = tensor::new_1d(units);
	x_grad = tensor::new_1d(units);
	y = tensor::new_1d(internal_rh->y_units);
	y_grad = tensor::new_1d(internal_rh->y_units);
	a = tensor::new_1d(units);
	a_grad = tensor::new_1d(units);
	e = tensor::new_1d(units);
	e_grad = tensor::new_1d(units);
	internal_rh->compile();
	a_model->compile();
	e_model->compile();

	x.group_join_all_ranks(internal_rh->x);
	x.group_join_all_ranks(a_model->x);
	x.group_join_all_ranks(e_model->x);

	a.group_join_all_ranks(a_model->y);
	a_grad.group_join_all_ranks(a_model->y_grad);
	e.group_join_all_ranks(e_model->y);
	e_grad.group_join_all_ranks(e_model->y_grad);

	internal_rh->y.group_join_all_ranks(y);
	internal_rh->y_grad.group_join_all_ranks(y_grad);
}