#include "pch.h"
#include "ntm_ts.h"

using aurora::models::ntm_ts;

ntm_ts::~ntm_ts() {

}

ntm_ts::ntm_ts() {

}

ntm_ts::ntm_ts(
	size_t a_memory_height,
	size_t a_memory_width,
	size_t a_num_readers,
	size_t a_num_writers,
	vector<int> a_valid_shifts,
	vector<size_t> a_head_hidden_dims,
	function<void(ptr<param>&)> a_func) {
	memory_height = a_memory_height;
	memory_width = a_memory_width;

	for (int i = 0; i < a_num_readers; i++)
		internal_readers.push_back(new ntm_reader(a_memory_height, a_memory_width, a_valid_shifts, a_head_hidden_dims, a_func));
	for (int i = 0; i < a_num_writers; i++)
		internal_writers.push_back(new ntm_writer(a_memory_height, a_memory_width, a_valid_shifts, a_head_hidden_dims, a_func));

}

void ntm_ts::pmt_wise(function<void(ptr<param>&)> a_func) {
	for (int i = 0; i < internal_readers.size(); i++)
		internal_readers[i]->pmt_wise(a_func);
	for (int i = 0; i < internal_writers.size(); i++)
		internal_writers[i]->pmt_wise(a_func);
}

model* ntm_ts::clone() {
	ntm_ts* result = new ntm_ts();
	result->memory_height = memory_height;
	result->memory_width = memory_width;
	for (int i = 0; i < internal_readers.size(); i++)
		result->internal_readers.push_back((ntm_reader*)internal_readers[i]->clone());
	for (int i = 0; i < internal_writers.size(); i++)
		result->internal_writers.push_back((ntm_writer*)internal_writers[i]->clone());
	return result;
}

model* ntm_ts::clone(function<void(ptr<param>&)> a_func) {
	ntm_ts* result = new ntm_ts();
	result->memory_height = memory_height;
	result->memory_width = memory_width;
	for (int i = 0; i < internal_readers.size(); i++)
		result->internal_readers.push_back((ntm_reader*)internal_readers[i]->clone(a_func));
	for (int i = 0; i < internal_writers.size(); i++)
		result->internal_writers.push_back((ntm_writer*)internal_writers[i]->clone(a_func));
	return result;
}

void ntm_ts::fwd() {
	for (int i = 0; i < internal_writers.size(); i++)
		internal_writers[i]->fwd();
	y.clear();
	for (int i = 0; i < internal_readers.size(); i++) {
		internal_readers[i]->fwd();
		y.add_1d(internal_readers[i]->y, y);
	}
}

void ntm_ts::bwd() {
	x_grad.clear();
	accum_my_grad.pop(my_grad);
	y_grad.add_1d(hty_grad, reader_y_grad);
	for (int i = 0; i < internal_readers.size(); i++) {
		internal_readers[i]->bwd();
		x_grad.add_1d(internal_readers[i]->x_grad, x_grad);
		accum_my_grad.add_2d(internal_readers[i]->mx_grad, accum_my_grad);
	}
	for (int i = 0; i < internal_writers.size(); i++) {
		internal_writers[i]->bwd();
		x_grad.add_1d(internal_writers[i]->x_grad, x_grad);
	}
}

tensor& ntm_ts::fwd(tensor& a_x) {
	x.pop(a_x);
	fwd();
	return y;
}

tensor& ntm_ts::bwd(tensor& a_y_grad) {
	y_grad.pop(a_y_grad);
	bwd();
	return x_grad;
}

void ntm_ts::signal(tensor& a_y_des) {
	y.sub_1d(a_y_des, y_grad);
}

void ntm_ts::cycle(tensor& a_x, tensor& a_y_des) {
	x.pop(a_x);
	fwd();
	signal(a_y_des);
	bwd();
}

void ntm_ts::recur(function<void(model*)> a_func) {
	a_func(this);
	for (int i = 0; i < internal_readers.size(); i++)
		internal_readers[i]->recur(a_func);
	for (int i = 0; i < internal_writers.size(); i++)
		internal_writers[i]->recur(a_func);
}

void ntm_ts::compile() {
	x = tensor::new_1d(memory_width);
	x_grad = tensor::new_1d(memory_width);
	y = tensor::new_1d(memory_width);
	y_grad = tensor::new_1d(memory_width);
	mx = tensor::new_2d(memory_height, memory_width);
	mx_grad = tensor::new_2d(memory_height, memory_width);
	my = tensor::new_2d(memory_height, memory_width);
	my_grad = tensor::new_2d(memory_height, memory_width);
	reader_y_grad = tensor::new_1d(memory_width);
	accum_my_grad = tensor::new_2d(memory_height, memory_width);

	tensor* l_mx = &mx;
	tensor* l_mx_grad = &mx_grad;

	for (int i = 0; i < internal_writers.size(); i++) {
		internal_writers[i]->compile();
		internal_writers[i]->x.group_join_all_ranks(x);
		internal_writers[i]->mx.group_join_all_ranks(*l_mx);
		internal_writers[i]->mx_grad.group_join_all_ranks(*l_mx_grad);
		l_mx = &internal_writers[i]->y;
		l_mx_grad = &internal_writers[i]->y_grad;
	}

	l_mx->group_join_all_ranks(my);
	l_mx_grad->group_join_all_ranks(accum_my_grad);

	for (int i = 0; i < internal_readers.size(); i++) {
		internal_readers[i]->compile();
		internal_readers[i]->x.group_join_all_ranks(x);
		internal_readers[i]->mx.group_join_all_ranks(my);
		internal_readers[i]->y_grad.group_join_all_ranks(reader_y_grad);
	}

}