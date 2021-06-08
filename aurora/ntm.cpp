#include "pch.h"
#include "ntm.h"

using aurora::models::ntm;

ntm::~ntm() {

}

ntm::ntm() {

}

ntm::ntm(
	size_t a_memory_height,
	size_t a_memory_width,
	size_t a_num_readers,
	size_t a_num_writers,
	vector<int> a_valid_shifts,
	vector<size_t> a_head_hidden_dims,
	function<void(ptr<param>&)> a_func) {
	memory_height = a_memory_height;
	memory_width = a_memory_width;
	
	internal_lstm = new lstm(a_memory_width, a_func);

	ntm_ts_template = new ntm_ts(
		a_memory_height,
		a_memory_width,
		a_num_readers,
		a_num_writers,
		a_valid_shifts,
		a_head_hidden_dims,
		a_func);

}

void ntm::pmt_wise(function<void(ptr<param>&)> a_func) {
	internal_lstm->pmt_wise(a_func);
	ntm_ts_template->pmt_wise(a_func);
}

model* ntm::clone() {
	ntm* result = new ntm();
	result->memory_height = memory_height;
	result->memory_width = memory_width;
	result->internal_lstm = (lstm*)internal_lstm->clone();
	result->ntm_ts_template = (ntm_ts*)ntm_ts_template->clone();
	result->prep(prepared.size());
	result->unroll(prepared.size());
	return result;
}

model* ntm::clone(function<void(ptr<param>&)> a_func) {
	ntm* result = new ntm();
	result->memory_height = memory_height;
	result->memory_width = memory_width;
	result->internal_lstm = (lstm*)internal_lstm->clone();
	result->ntm_ts_template = (ntm_ts*)ntm_ts_template->clone(a_func);
	result->prep(prepared.size());
	result->unroll(prepared.size());
	return result;
}

void ntm::fwd() {
	for (int i = 0; i < unrolled.size(); i++) {
		internal_lstm->unrolled[i]->fwd();
		unrolled[i]->fwd();
		internal_lstm->unrolled[i]->hty.add_1d(unrolled[i]->y, internal_lstm->unrolled[i]->hty);
	}
}

void ntm::bwd() {
	for (int i = unrolled.size() - 1; i >= 0; i--) {
		unrolled[i]->bwd();
		internal_lstm->unrolled[i]->bwd();
	}
}

tensor& ntm::fwd(tensor& a_x) {
	x.pop(a_x);
	fwd();
	return y;
}

tensor& ntm::bwd(tensor& a_y_grad) {
	y_grad.pop(a_y_grad);
	bwd();
	return x_grad;
}

void ntm::signal(tensor& a_y_des) {
	y.sub_2d(a_y_des, y_grad);
}

void ntm::cycle(tensor& a_x, tensor& a_y_des) {
	x.pop(a_x);
	fwd();
	signal(a_y_des);
	bwd();
}

void ntm::recur(function<void(model*)> a_func) {
	a_func(this);
	internal_lstm->recur(a_func);
	ntm_ts_template->recur(a_func);
}

void ntm::compile() {
	x = tensor::new_2d(prepared.size(), memory_width);
	x_grad = tensor::new_2d(prepared.size(), memory_width);
	y = tensor::new_2d(prepared.size(), memory_width);
	y_grad = tensor::new_2d(prepared.size(), memory_width);
	mx = tensor::new_2d(memory_height, memory_width, 1.0);	
	mx_grad = tensor::new_2d(memory_height, memory_width);
	my = tensor::new_2d(memory_height, memory_width);
	my_grad = tensor::new_2d(memory_height, memory_width);

	internal_lstm->compile();
	x.group_join_all_ranks(internal_lstm->x);
	x_grad.group_join_all_ranks(internal_lstm->x_grad);

	tensor* l_mx = &mx;
	tensor* l_mx_grad = &mx_grad;

	for (int i = 0; i < prepared.size(); i++) {
		
		prepared[i]->compile();
		internal_lstm->y[i].group_join(prepared[i]->x);
		internal_lstm->y_grad[i].group_join(prepared[i]->x_grad);
		prepared[i]->y.group_join(y[i]);
		prepared[i]->y_grad.group_join(y_grad[i]);
		prepared[i]->hty_grad.group_join(internal_lstm->prepared[i]->hty_grad);
		
		prepared[i]->mx.group_join(*l_mx);
		l_mx_grad->group_join(prepared[i]->mx_grad);

		l_mx = &prepared[i]->my;
		l_mx_grad = &prepared[i]->my_grad;

		if (i > 0) {
			for (int j = 0; j < prepared[i]->internal_readers.size(); j++) {
				prepared[i]->internal_readers[j]->wx.group_join(prepared[i - 1]->internal_readers[j]->wy);
				prepared[i]->internal_readers[j]->wx_grad.group_join(prepared[i - 1]->internal_readers[j]->wy_grad);
			}
			for (int j = 0; j < prepared[i]->internal_writers.size(); j++) {
				prepared[i]->internal_writers[j]->wx.group_join(prepared[i - 1]->internal_writers[j]->wy);
				prepared[i]->internal_writers[j]->wx_grad.group_join(prepared[i - 1]->internal_writers[j]->wy_grad);
			}
		}
	}

	l_mx->group_join(my);
	l_mx_grad->group_join(my_grad);

}

void ntm::prep(size_t a_n) {
	internal_lstm->prep(a_n);
	prepared.clear();
	prepared.resize(a_n);
	for (int i = 0; i < a_n; i++)
		prepared.at(i) = (ntm_ts*)ntm_ts_template->clone();
}

void ntm::unroll(size_t a_n) {
	internal_lstm->unroll(a_n);
	unrolled.clear();
	unrolled.resize(a_n);
	for (int i = 0; i < a_n; i++)
		unrolled.at(i) = prepared.at(i);
}