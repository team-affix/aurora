#include "pch.h"
#include "ntm_writer.h"

using aurora::models::ntm_writer;

ntm_writer::~ntm_writer() {

}

ntm_writer::ntm_writer() {
	memory_height = 0;
	memory_width = 0;
}

ntm_writer::ntm_writer(
	size_t a_memory_height,
	size_t a_memory_width,
	vector<size_t> a_head_h_dims,
	vector<int> a_valid_shifts,
	function<void(ptr<param>&)> a_func) {
	memory_height = a_memory_height;
	memory_width = a_memory_width;
	head = new ntm_wh(a_memory_width, a_head_h_dims, a_valid_shifts.size(), a_func);
	addresser = new ntm_addresser(a_memory_height, a_memory_width, a_valid_shifts);
}

void ntm_writer::pmt_wise(function<void(ptr<param>&)> a_func) {
	head->pmt_wise(a_func);
	addresser->pmt_wise(a_func);
}

model* ntm_writer::clone() {
	ntm_writer* result = new ntm_writer();
	result->memory_height = memory_height;
	result->memory_width = memory_width;
	result->head = (ntm_wh*)head->clone();
	result->addresser = (ntm_addresser*)addresser->clone();
	return result;
}

model* ntm_writer::clone(function<void(ptr<param>&)> a_func) {
	ntm_writer* result = new ntm_writer();
	result->memory_height = memory_height;
	result->memory_width = memory_width;
	result->head = (ntm_wh*)head->clone(a_func);
	result->addresser = (ntm_addresser*)addresser->clone(a_func);
	return result;
}

void ntm_writer::fwd() {
	head->fwd();
	addresser->fwd();

	tensor& weighting = addresser->y;

	// WRITE TO MEMORY
	for(int i = 0; i < memory_height; i++)
		for (int j = 0; j < memory_width; j++)
			y[i][j].val() = mx[i][j] * (1 - weighting[i] * head->e[j]) + weighting[i] * head->a[j];

}

void ntm_writer::bwd() {

	addresser->y_grad.clear();
	head->e_grad.clear();
	head->a_grad.clear();

	// CALCULATE GRADS
	for (int i = 0; i < memory_height; i++)
		for (int j = 0; j < memory_width; j++) {
			mx_grad[i][j].val() = y_grad[i][j] * (1.0 - addresser->y[i] * head->e[j]);
			addresser->y_grad[i].val() += y_grad[i][j] * mx[i][j] * -head->e[j];
			head->e_grad[j].val() += y_grad[i][j] * mx[i][j] * -addresser->y[i];
			head->a_grad[j].val() += y_grad[i][j] * addresser->y[i];
		}

	addresser->bwd();
	head->bwd();

}

tensor& ntm_writer::fwd(tensor& a_x) {
	x.pop(a_x);
	fwd();
	return y;
}

tensor& ntm_writer::bwd(tensor& a_y_grad) {
	y_grad.pop(a_y_grad);
	bwd();
	return x_grad;
}

void ntm_writer::signal(tensor& a_y_des) {
	y.sub_2d(a_y_des, y_grad);
}

void ntm_writer::cycle(tensor& a_x, tensor& a_y_des) {
	x.pop(a_x);
	fwd();
	signal(a_y_des);
	bwd();
}

void ntm_writer::recur(function<void(model*)> a_func) {
	head->recur(a_func);
	addresser->recur(a_func);
}

void ntm_writer::compile() {
	x = tensor::new_1d(memory_width);
	x_grad = tensor::new_1d(memory_width);
	y = tensor::new_2d(memory_height, memory_width);
	y_grad = tensor::new_2d(memory_height, memory_width);

	mx = tensor::new_2d(memory_height, memory_width);
	mx_grad = tensor::new_2d(memory_height, memory_width);

	head->compile();
	addresser->compile();

	x.group_join(head->x);
	x_grad.group_join(head->x_grad);
	addresser->x.group_join(head->y);
	addresser->x_grad.group_join(head->y_grad);
	addresser->mx.group_join(mx);
	addresser->mx_grad.group_join(mx_grad);
}