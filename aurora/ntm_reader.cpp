#include "pch.h"
#include "ntm_reader.h"

using aurora::models::ntm_reader;

ntm_reader::~ntm_reader() {

}

ntm_reader::ntm_reader() {
	memory_height = 0;
	memory_width = 0;
}

ntm_reader::ntm_reader(
	size_t a_memory_height,
	size_t a_memory_width,
	vector<size_t> a_head_h_dims,
	vector<int> a_valid_shifts,
	function<void(ptr<param>&)> a_func) {
	memory_height = a_memory_height;
	memory_width = a_memory_width;
	head = new ntm_rh(a_memory_width, a_head_h_dims, a_valid_shifts.size(), a_func);
	addresser = new ntm_addresser(a_memory_height, a_memory_width, a_valid_shifts);
}

void ntm_reader::pmt_wise(function<void(ptr<param>&)> a_func) {
	head->pmt_wise(a_func);
	addresser->pmt_wise(a_func);
}

model* ntm_reader::clone() {
	ntm_reader* result = new ntm_reader();
	result->memory_height = memory_height;
	result->memory_width = memory_width;
	result->head = (ntm_rh*)head->clone();
	result->addresser = (ntm_addresser*)addresser->clone();
	return result;
}

model* ntm_reader::clone(function<void(ptr<param>&)> a_func) {
	ntm_reader* result = new ntm_reader();
	result->memory_height = memory_height;
	result->memory_width = memory_width;
	result->head = (ntm_rh*)head->clone(a_func);
	result->addresser = (ntm_addresser*)addresser->clone(a_func);
	return result;
}

void ntm_reader::fwd() {
	head->fwd();
	addresser->fwd();

	// APPLY WEIGHTING
	for (int i = 0; i < memory_height; i++)
		for (int j = 0; j < memory_width; j++)
			mx_weighted[i][j].val() = mx[i][j] * addresser->y[i];

	mx_weighted.sum_2d(y);

}

void ntm_reader::bwd() {

	addresser->y_grad.clear();

	// CALCULATE GRADS
	for(int i = 0; i < memory_height; i++)
		for (int j = 0; j < memory_width; j++) {
			mx_grad[i][j].val() = y_grad[j] * addresser->y[i];
			addresser->y_grad[i].val() += y_grad[j] * mx[i][j];
		}

	addresser->bwd();
	head->bwd();
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
	head->recur(a_func);
	addresser->recur(a_func);
}

void ntm_reader::compile() {

	x = tensor::new_1d(memory_width);
	x_grad = tensor::new_1d(memory_width);
	y = tensor::new_1d(memory_width);
	y_grad = tensor::new_1d(memory_width);

	mx = tensor::new_2d(memory_height, memory_width);
	mx_grad = tensor::new_2d(memory_height, memory_width);
	mx_weighted = tensor::new_2d(memory_height, memory_width);

	head->compile();
	addresser->compile();

	x.group_join(head->x);
	x_grad.group_join(head->x_grad);
	addresser->x.group_join(head->y);
	addresser->x_grad.group_join(head->y_grad);
	addresser->mx.group_join(mx);
	addresser->mx_grad.group_join(mx_grad);

}