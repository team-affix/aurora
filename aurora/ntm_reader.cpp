#include "pch.h"
#include "ntm_reader.h"

using aurora::models::ntm_reader;

ntm_reader::~ntm_reader() {

}

ntm_reader::ntm_reader() {

}

ntm_reader::ntm_reader(size_t a_memory_height, size_t a_memory_width, vector<int> a_valid_shifts, vector<size_t> a_head_hidden_dims, function<void(ptr<param>&)> a_func) {
	memory_height = a_memory_height;
	memory_width = a_memory_width;
	internal_head = new ntm_rh(a_memory_width, a_head_hidden_dims, a_valid_shifts.size(), a_func);
	internal_addresser = new ntm_addresser(a_memory_height, a_memory_width, a_valid_shifts);
}

void ntm_reader::pmt_wise(function<void(ptr<param>&)> a_func) {
	internal_head->pmt_wise(a_func);
	internal_addresser->pmt_wise(a_func);
}

model* ntm_reader::clone() {
	ntm_reader* result = new ntm_reader();
	result->memory_height = memory_height;
	result->memory_width = memory_width;
	result->internal_head = (ntm_rh*)internal_head->clone();
	result->internal_addresser = (ntm_addresser*)internal_addresser->clone();
	return result;
}

model* ntm_reader::clone(function<void(ptr<param>&)> a_func) {
	ntm_reader* result = new ntm_reader();
	result->memory_height = memory_height;
	result->memory_width = memory_width;
	result->internal_head = (ntm_rh*)internal_head->clone(a_func);
	result->internal_addresser = (ntm_addresser*)internal_addresser->clone(a_func);
	return result;
}

void ntm_reader::fwd() {
	internal_head->fwd();
	internal_addresser->fwd();
	y.clear();
	for (int i = 0; i < memory_height; i++)
		for (int j = 0; j < memory_width; j++)
			y[j].val() += mx[i][j] * internal_addresser->y[i];
}

void ntm_reader::bwd() {
	internal_addresser->y_grad.clear();
	for (int i = 0; i < memory_height; i++)
		for (int j = 0; j < memory_width; j++) {
			internal_addresser->y_grad[i].val() += y_grad[j] * mx[i][j];
			mx_grad[i][j].val() = y_grad[j] * internal_addresser->y[i];
		}
	internal_addresser->bwd();
	internal_head->bwd();
}

tensor& ntm_reader::fwd(tensor& a_x) {
	x.pop(a_x);
	fwd();
	return y;
}

tensor& ntm_reader::bwd(tensor& a_y_grad) {
	y_grad.pop(a_y_grad);
	bwd();
	return x_grad;
}

void ntm_reader::signal(tensor& a_y_des) {
	y.sub_1d(a_y_des, y_grad);
}

void ntm_reader::cycle(tensor& a_x, tensor& a_y_des) {
	x.pop(a_x);
	fwd();
	signal(a_y_des);
	bwd();
}

void ntm_reader::recur(function<void(model*)> a_func) {
	a_func(this);
	internal_head->recur(a_func);
	internal_addresser->recur(a_func);
}

void ntm_reader::compile() {
	x = tensor::new_1d(memory_width);
	x_grad = tensor::new_1d(memory_width);
	y = tensor::new_1d(memory_width);
	y_grad = tensor::new_1d(memory_width);
	mx = tensor::new_2d(memory_height, memory_width);
	mx_grad = tensor::new_2d(memory_height, memory_width);
	wx = tensor::new_1d(memory_height);
	wx_grad = tensor::new_1d(memory_height);
	wy = tensor::new_1d(memory_height);
	wy_grad = tensor::new_1d(memory_height);

	internal_head->compile();
	internal_addresser->compile();

	internal_head->k.group_join(internal_addresser->key);
	internal_head->k_grad.group_join(internal_addresser->key_grad);
	internal_head->beta.group_join(internal_addresser->beta);
	internal_head->beta_grad.group_join(internal_addresser->beta_grad);
	internal_head->g.group_join(internal_addresser->g);
	internal_head->g_grad.group_join(internal_addresser->g_grad);
	internal_head->s.group_join(internal_addresser->s);
	internal_head->s_grad.group_join(internal_addresser->s_grad);
	internal_head->gamma.group_join(internal_addresser->gamma);
	internal_head->gamma_grad.group_join(internal_addresser->gamma_grad);

	x.group_join_all_ranks(internal_head->x);
	x_grad.group_join_all_ranks(internal_head->x_grad);
	mx.group_join_all_ranks(internal_addresser->x);
	mx_grad.group_join_all_ranks(internal_addresser->x_grad);
	wx.group_join_all_ranks(internal_addresser->wx);
	wx_grad.group_join_all_ranks(internal_addresser->wx_grad);
	internal_addresser->wy.group_join_all_ranks(wy);
	internal_addresser->wy_grad.group_join_all_ranks(wy_grad);

}