#include "pch.h"
#include "cnl.h"
#include "weight_junction.h"

using aurora::models::cnl;

cnl::~cnl() {

}

cnl::cnl() {

}

cnl::cnl(size_t a_input_max_height, size_t a_input_max_width, size_t a_filter_height, size_t a_filter_width, size_t a_stride_len, function<void(ptr<param>&)> a_func) {
	this->filter_height = a_filter_height;
	this->filter_width = a_filter_width;
	this->stride_len = a_stride_len;
	filter_template = new weight_junction(filter_height * filter_width, 1, a_func);
	filters = new sync(new sync(filter_template));
	prep(a_input_max_height, a_input_max_width);
}

cnl::cnl(size_t a_input_max_height, size_t a_input_max_width, size_t a_filter_height, size_t a_filter_width, size_t a_stride_len, ptr<model> a_filter_template) {
	this->filter_height = a_filter_height;
	this->filter_width = a_filter_width;
	this->stride_len = a_stride_len;
	filter_template = a_filter_template;
	filters = new sync(new sync(filter_template));
	prep(a_input_max_height, a_input_max_width);
}

void cnl::pmt_wise(function<void(ptr<param>&)> a_func) {
	filter_template->pmt_wise(a_func);
}

model* cnl::clone() {
	cnl* result = new cnl(input_max_height, input_max_width, filter_height, filter_width, stride_len, filter_template->clone());
	return result;
}

model* cnl::clone(function<void(ptr<param>&)> a_func) {
	cnl* result = new cnl(input_max_height, input_max_width, filter_height, filter_width, stride_len, filter_template->clone(a_func));
	return result;
}

void cnl::fwd() {
	filters->fwd();
}

void cnl::bwd() {
	filters->bwd();
}

tensor& cnl::fwd(tensor& a_x) {
	x.pop(a_x);
	fwd();
	return y;
}

tensor& cnl::bwd(tensor& a_y_grad) {
	y_grad.pop(a_y_grad);
	bwd();
	return x_grad;
}

void cnl::signal(tensor& a_y_des) {
	y_des.pop(a_y_des);
	filters->signal(y_des_reshaped);
}

void cnl::cycle(tensor& a_x, tensor& a_y_des) {
	x.pop(a_x);
	fwd();
	signal(a_y_des);
	bwd();
}

void cnl::recur(function<void(model*)> a_func) {
	a_func(this);
	filter_template->recur(a_func);
}

void cnl::compile() {
	this->x = tensor::new_2d(input_max_height, input_max_width);
	this->x_grad = tensor::new_2d(input_max_height, input_max_width);
	this->y = tensor::new_2d(y_strides(input_max_height), x_strides(input_max_width));
	this->y_grad = tensor::new_2d(y_strides(input_max_height), x_strides(input_max_width));
	this->y_des = tensor::new_2d(y_strides(input_max_height), x_strides(input_max_width));
	this->y_des_reshaped = tensor::new_2d(y_strides(input_max_height), x_strides(input_max_width), { 0 });

	// JOIN RESHAPED_Y TO Y
	for(int i = 0; i < y_des.height(); i++)
		for (int j = 0; j < y_des.width(); j++)
			y_des_reshaped[i][j][0].group_join(y_des[i][j]);

	filters->compile();
	for (int i = 0; i < y_strides(input_max_height); i++) {
		int y_start = i * stride_len;
		sync* row = (sync*)filters->prepared[i].get();
		for (int j = 0; j < x_strides(input_max_width); j++) {
			int x_start = j * stride_len;
			ptr<model>& filter = row->prepared[j];
			tensor filter_range_x = x.range_2d(y_start, x_start, filter_height, filter_width).unroll();
			tensor filter_range_x_grad = x_grad.range_2d(y_start, x_start, filter_height, filter_width).unroll();
			tensor filter_range_y = y.range_2d(i, j, 1, 1).unroll();
			tensor filter_range_y_grad = y_grad.range_2d(i, j, 1, 1).unroll();
			filter_range_x.group_join_all_ranks(filter->x);
			filter_range_x_grad.group_join_all_ranks(filter->x_grad);
			filter_range_y.group_join_all_ranks(filter->y);
			filter_range_y_grad.group_join_all_ranks(filter->y_grad);
		}
	}
}

void cnl::prep(size_t a_a, size_t a_b) {
	this->input_max_height = a_a;
	this->input_max_width = a_b;
	sync* row_sync = (sync*)filters->model_template.get();
	row_sync->prep(x_strides(a_b));
	filters->prep(y_strides(a_a));
}

void cnl::unroll(size_t a_a, size_t a_b) {
	size_t x_filters = x_strides(a_b);
	size_t y_filters = y_strides(a_a);
	filters->unroll(y_filters);
	for (int i = 0; i < filters->unrolled.size(); i++) {
		sync* row_sync = (sync*)filters->unrolled[i].get();
		row_sync->unroll(x_filters);
	}
}

size_t cnl::x_strides(size_t a_width) {
	return (a_width - filter_width) / stride_len + 1;
}

size_t cnl::x_strides() {
	return (input_max_width - filter_width) / stride_len + 1;
}

size_t cnl::y_strides(size_t a_height) {
	return (a_height - filter_height) / stride_len + 1;
}

size_t cnl::y_strides() {
	return (input_max_height - filter_height) / stride_len + 1;
}