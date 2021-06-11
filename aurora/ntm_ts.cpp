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
	function<void(Param&)> a_func) {
	memory_height = a_memory_height;
	memory_width = a_memory_width;

	for (int i = 0; i < a_num_readers; i++)
		internal_readers.push_back(new ntm_reader(a_memory_height, a_memory_width, a_valid_shifts, a_head_hidden_dims, a_func));
	for (int i = 0; i < a_num_writers; i++)
		internal_writers.push_back(new ntm_writer(a_memory_height, a_memory_width, a_valid_shifts, a_head_hidden_dims, a_func));

}

void ntm_ts::param_recur(function<void(Param&)> a_func) {
	for (int i = 0; i < internal_readers.size(); i++)
		internal_readers[i]->param_recur(a_func);
	for (int i = 0; i < internal_writers.size(); i++)
		internal_writers[i]->param_recur(a_func);
}

model* ntm_ts::clone(function<Param(Param&)> a_func) {
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

void ntm_ts::signal(tensor& a_y_des) {
	y.sub_1d(a_y_des, y_grad);
}

void ntm_ts::model_recur(function<void(model*)> a_func) {
	a_func(this);
	for (int i = 0; i < internal_readers.size(); i++)
		internal_readers[i]->model_recur(a_func);
	for (int i = 0; i < internal_writers.size(); i++)
		internal_writers[i]->model_recur(a_func);
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

	// CREATE TENSORS FOR OUTERMOST WEIGHTINGS (wx,wy)
	read_wx = tensor::new_2d(internal_readers.size(), memory_height);
	read_wx_grad = tensor::new_2d(internal_readers.size(), memory_height);
	read_wy = tensor::new_2d(internal_readers.size(), memory_height);
	read_wy_grad = tensor::new_2d(internal_readers.size(), memory_height);
	write_wx = tensor::new_2d(internal_writers.size(), memory_height);
	write_wx_grad = tensor::new_2d(internal_writers.size(), memory_height);
	write_wy = tensor::new_2d(internal_writers.size(), memory_height);
	write_wy_grad = tensor::new_2d(internal_writers.size(), memory_height);

	tensor* l_mx = &mx;
	tensor* l_mx_grad = &mx_grad;

	for (int i = 0; i < internal_writers.size(); i++) {
		internal_writers[i]->compile();
		internal_writers[i]->x.group_join_all_ranks(x);
		internal_writers[i]->mx.group_join_all_ranks(*l_mx);
		internal_writers[i]->mx_grad.group_join_all_ranks(*l_mx_grad);
		l_mx = &internal_writers[i]->y;
		l_mx_grad = &internal_writers[i]->y_grad;

		write_wx[i].group_join(internal_writers[i]->wx);
		write_wx_grad[i].group_join(internal_writers[i]->wx_grad);
		write_wy[i].group_join(internal_writers[i]->wy);
		write_wy_grad[i].group_join(internal_writers[i]->wy_grad);
	}

	l_mx->group_join_all_ranks(my);
	l_mx_grad->group_join_all_ranks(accum_my_grad);

	for (int i = 0; i < internal_readers.size(); i++) {
		internal_readers[i]->compile();
		internal_readers[i]->x.group_join_all_ranks(x);
		internal_readers[i]->mx.group_join_all_ranks(my);
		internal_readers[i]->y_grad.group_join_all_ranks(reader_y_grad);

		read_wx[i].group_join(internal_readers[i]->wx);
		read_wx_grad[i].group_join(internal_readers[i]->wx_grad);
		read_wy[i].group_join(internal_readers[i]->wy);
		read_wy_grad[i].group_join(internal_readers[i]->wy_grad);
	}

}