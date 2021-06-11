#include "pch.h"
#include "ntm_writer.h"

using aurora::models::ntm_writer;

ntm_writer::~ntm_writer() {

}

ntm_writer::ntm_writer() {

}

ntm_writer::ntm_writer(size_t a_memory_height, size_t a_memory_width, vector<int> a_valid_shifts, vector<size_t> a_head_hidden_dims, function<void(ptr<param>&)> a_func) {
	memory_height = a_memory_height;
	memory_width = a_memory_width;
	internal_head = new ntm_wh(a_memory_width, a_head_hidden_dims, a_valid_shifts.size(), a_func);
	internal_addresser = new ntm_addresser(a_memory_height, a_memory_width, a_valid_shifts);
}

void ntm_writer::pmt_wise(function<void(ptr<param>&)> a_func) {
	internal_head->pmt_wise(a_func);
	internal_addresser->pmt_wise(a_func);
}

model* ntm_writer::clone() {
	ntm_writer* result = new ntm_writer();
	result->memory_height = memory_height;
	result->memory_width = memory_width;
	result->internal_head = (ntm_wh*)internal_head->clone();
	result->internal_addresser = (ntm_addresser*)internal_addresser->clone();
	return result;
}

model* ntm_writer::clone(function<void(ptr<param>&)> a_func) {
	ntm_writer* result = new ntm_writer();
	result->memory_height = memory_height;
	result->memory_width = memory_width;
	result->internal_head = (ntm_wh*)internal_head->clone(a_func);
	result->internal_addresser = (ntm_addresser*)internal_addresser->clone(a_func);
	return result;
}

void ntm_writer::fwd() {
	internal_head->fwd();
	internal_addresser->fwd();
	for (int i = 0; i < memory_height; i++)
		for (int j = 0; j < memory_width; j++) {
			weighted_erase_compliment[i][j].val() = 1.0 - internal_addresser->y[i] * internal_head->e[j];
			y[i][j].val() =
				mx[i][j] * weighted_erase_compliment[i][j];
			y[i][j].val() +=
				internal_addresser->y[i] *
				internal_head->a[j];
		}
}

void ntm_writer::bwd() {
	internal_addresser->y_grad.clear();
	internal_head->e_grad.clear();
	internal_head->a_grad.clear();
	for (int i = 0; i < memory_height; i++)
		for (int j = 0; j < memory_width; j++) {
			mx_grad[i][j].val() = y_grad[i][j] * weighted_erase_compliment[i][j];
			internal_addresser->y_grad[i].val() += y_grad[i][j] * 
				(-mx[i][j] * internal_head->e[j] + internal_head->a[j]);
			internal_head->e_grad[j].val() += y_grad[i][j] *
				-mx[i][j] * internal_addresser->y[i];
			internal_head->a_grad[j].val() += y_grad[i][j] *
				internal_addresser->y[i];
		}
	internal_addresser->bwd();
	mx_grad.add_2d(internal_addresser->x_grad, mx_grad);
	internal_head->bwd();
}

void ntm_writer::signal(tensor& a_y_des) {
	y.sub_2d(a_y_des, y_grad);
}

void ntm_writer::recur(function<void(model*)> a_func) {
	a_func(this);
	internal_head->recur(a_func);
	internal_addresser->recur(a_func);
}

void ntm_writer::compile() {
	x = tensor::new_1d(memory_width);
	x_grad = tensor::new_1d(memory_width);
	y = tensor::new_2d(memory_height, memory_width);
	y_grad = tensor::new_2d(memory_height, memory_width);
	weighted_erase_compliment = tensor::new_2d(memory_height, memory_width);
	mx = tensor::new_2d(memory_height, memory_width);
	mx_grad = tensor::new_2d(memory_height, memory_width);
	wx = tensor::new_1d(memory_height);
	wx_grad = tensor::new_1d(memory_height);
	wy = tensor::new_1d(memory_height);
	wy_grad = tensor::new_1d(memory_height);

	internal_head->compile();
	internal_addresser->compile();

	internal_head->internal_rh->key.group_join(internal_addresser->key);
	internal_head->internal_rh->key_grad.group_join(internal_addresser->key_grad);
	internal_head->internal_rh->beta.group_join(internal_addresser->beta);
	internal_head->internal_rh->beta_grad.group_join(internal_addresser->beta_grad);
	internal_head->internal_rh->g.group_join(internal_addresser->g);
	internal_head->internal_rh->g_grad.group_join(internal_addresser->g_grad);
	internal_head->internal_rh->s.group_join(internal_addresser->s);
	internal_head->internal_rh->s_grad.group_join(internal_addresser->s_grad);
	internal_head->internal_rh->gamma.group_join(internal_addresser->gamma);
	internal_head->internal_rh->gamma_grad.group_join(internal_addresser->gamma_grad);

	x.group_join_all_ranks(internal_head->x);
	x_grad.group_join_all_ranks(internal_head->x_grad);
	mx.group_join_all_ranks(internal_addresser->x);
	wx.group_join_all_ranks(internal_addresser->wx);
	wx_grad.group_join_all_ranks(internal_addresser->wx_grad);
	internal_addresser->wy.group_join_all_ranks(wy);
	internal_addresser->wy_grad.group_join_all_ranks(wy_grad);
}