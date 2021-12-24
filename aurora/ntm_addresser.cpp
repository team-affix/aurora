#include "affix-base/pch.h"
#include "ntm_addresser.h"

using aurora::models::ntm_addresser;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::params::param_sgd;
using aurora::maths::tensor;
using std::vector;

ntm_addresser::~ntm_addresser() {

}

ntm_addresser::ntm_addresser() {

}

ntm_addresser::ntm_addresser(size_t a_memory_height, size_t a_memory_width, vector<int> a_valid_shifts) {
	memory_height = a_memory_height;
	memory_width = a_memory_width;
	shift_units = a_valid_shifts.size();
	internal_content_addresser = new ntm_content_addresser(a_memory_height, a_memory_width);
	internal_location_addresser = new ntm_location_addresser(a_memory_height, a_valid_shifts);
}

void ntm_addresser::param_recur(function<void(Param&)> a_func) {
	internal_content_addresser->param_recur(a_func);
	internal_location_addresser->param_recur(a_func);
}

model* ntm_addresser::clone(function<Param(Param&)> a_func) {
	ntm_addresser* result = new ntm_addresser();
	result->memory_height = memory_height;
	result->memory_width = memory_width;
	result->shift_units = shift_units;
	result->internal_content_addresser = (ntm_content_addresser*)internal_content_addresser->clone(a_func);
	result->internal_location_addresser = (ntm_location_addresser*)internal_location_addresser->clone(a_func);
	return result;
}

void ntm_addresser::fwd() {
	internal_content_addresser->fwd();
	internal_location_addresser->fwd();
}

void ntm_addresser::bwd() {
	internal_location_addresser->bwd();
	internal_content_addresser->bwd();
}

void ntm_addresser::signal(const tensor& a_y_des) {
	y.sub_1d(a_y_des, y_grad);
}

void ntm_addresser::model_recur(function<void(model*)> a_func) {
	a_func(this);
	internal_content_addresser->model_recur(a_func);
	internal_location_addresser->model_recur(a_func);
}

void ntm_addresser::compile() {
	x = tensor::new_2d(memory_height, memory_width);
	x_grad = tensor::new_2d(memory_height, memory_width);
	y = tensor::new_1d(memory_height);
	y_grad = tensor::new_1d(memory_height);
	key = tensor::new_1d(memory_width);
	key_grad = tensor::new_1d(memory_width);
	beta = tensor::new_1d(1);
	beta_grad = tensor::new_1d(1);
	wx = tensor::new_1d(memory_height);
	wx_grad = tensor::new_1d(memory_height);
	wy = tensor::new_1d(memory_height);
	wy_grad = tensor::new_1d(memory_height);
	g = tensor::new_1d(1);
	g_grad = tensor::new_1d(1);
	s = tensor::new_1d(shift_units);
	s_grad = tensor::new_1d(shift_units);
	gamma = tensor::new_1d(1);
	gamma_grad = tensor::new_1d(1);

	internal_content_addresser->compile();
	internal_location_addresser->compile();

	key.group_join(internal_content_addresser->key);
	key_grad.group_join(internal_content_addresser->key_grad);
	beta.group_join(internal_content_addresser->beta);
	beta_grad.group_join(internal_content_addresser->beta_grad);
	wx.group_join(internal_location_addresser->wx);
	wx_grad.group_join(internal_location_addresser->wx_grad);
	wy.group_join(internal_location_addresser->wy);
	wy_grad.group_join(internal_location_addresser->wy_grad);
	g.group_join(internal_location_addresser->g);
	g_grad.group_join(internal_location_addresser->g_grad);
	s.group_join(internal_location_addresser->s);
	s_grad.group_join(internal_location_addresser->s_grad);
	gamma.group_join(internal_location_addresser->gamma);
	gamma_grad.group_join(internal_location_addresser->gamma_grad);

	x.group_join(internal_content_addresser->x);
	x_grad.group_join(internal_content_addresser->x_grad);
	internal_content_addresser->y.group_join(internal_location_addresser->x);
	internal_content_addresser->y_grad.group_join(internal_location_addresser->x_grad);
	internal_location_addresser->y.group_join(y);
	internal_location_addresser->y_grad.group_join(y_grad);

}
