#include "affix-base/pch.h"
#include "cnl.h"
#include "weight_junction.h"

using aurora::models::cnl;
using std::function;
using aurora::params::Param;
using aurora::maths::tensor;
using aurora::models::model;

cnl::~cnl() {

}

cnl::cnl() {

}

cnl::cnl(size_t a_input_max_height, size_t a_input_max_width, size_t a_filter_height, size_t a_filter_width, size_t a_stride_len) {
	this->m_filter_height = a_filter_height;
	this->m_filter_width = a_filter_width;
	this->m_stride_len = a_stride_len;
	m_filter_template = new weight_junction(m_filter_height * m_filter_width, 1);
	m_filters = new sync(new sync(m_filter_template));
	prep(a_input_max_height, a_input_max_width);
}

cnl::cnl(size_t a_input_max_height, size_t a_input_max_width, size_t a_filter_height, size_t a_filter_width, size_t a_stride_len, Model a_filter_template) {
	this->m_filter_height = a_filter_height;
	this->m_filter_width = a_filter_width;
	this->m_stride_len = a_stride_len;
	m_filter_template = a_filter_template;
	m_filters = new sync(new sync(m_filter_template));
	prep(a_input_max_height, a_input_max_width);
}

void cnl::param_recur(function<void(Param&)> a_func) {
	m_filter_template->param_recur(a_func);
}

model* cnl::clone(function<Param(Param&)> a_func) {
	cnl* result = new cnl(m_input_max_height, m_input_max_width, m_filter_height, m_filter_width, m_stride_len, m_filter_template->clone(a_func));
	return result;
}

void cnl::fwd() {
	m_filters->fwd();
}

void cnl::bwd() {
	m_filters->bwd();
}

void cnl::signal(const tensor& a_y_des) {
	m_y_des.pop(a_y_des);
	m_filters->signal(m_y_des_reshaped);
}

void cnl::model_recur(function<void(model*)> a_func) {
	a_func(this);
	m_filter_template->model_recur(a_func);
}

void cnl::compile() {
	this->m_x = tensor::new_2d(m_input_max_height, m_input_max_width);
	this->m_x_grad = tensor::new_2d(m_input_max_height, m_input_max_width);
	this->m_y = tensor::new_2d(y_strides(m_input_max_height), x_strides(m_input_max_width));
	this->m_y_grad = tensor::new_2d(y_strides(m_input_max_height), x_strides(m_input_max_width));
	this->m_y_des = tensor::new_2d(y_strides(m_input_max_height), x_strides(m_input_max_width));
	this->m_y_des_reshaped = tensor::new_2d(y_strides(m_input_max_height), x_strides(m_input_max_width), { 0 });

	// JOIN RESHAPED_Y TO Y
	for(int i = 0; i < m_y_des.height(); i++)
		for (int j = 0; j < m_y_des.width(); j++)
			m_y_des_reshaped[i][j][0].group_join(m_y_des[i][j]);

	m_filters->compile();
	for (int i = 0; i < y_strides(m_input_max_height); i++) {
		int y_start = i * m_stride_len;
		sync* row = (sync*)m_filters->m_prepared[i].get();
		for (int j = 0; j < x_strides(m_input_max_width); j++) {
			int x_start = j * m_stride_len;
			Model& filter = row->m_prepared[j];
			tensor filter_range_x = m_x.range_2d(y_start, x_start, m_filter_height, m_filter_width).unroll();
			tensor filter_range_x_grad = m_x_grad.range_2d(y_start, x_start, m_filter_height, m_filter_width).unroll();
			tensor filter_range_y = m_y.range_2d(i, j, 1, 1).unroll();
			tensor filter_range_y_grad = m_y_grad.range_2d(i, j, 1, 1).unroll();
			filter_range_x.group_join_all_ranks(filter->m_x);
			filter_range_x_grad.group_join_all_ranks(filter->m_x_grad);
			filter_range_y.group_join_all_ranks(filter->m_y);
			filter_range_y_grad.group_join_all_ranks(filter->m_y_grad);
		}
	}
}

void cnl::prep(size_t a_a, size_t a_b) {
	this->m_input_max_height = a_a;
	this->m_input_max_width = a_b;
	sync* row_sync = (sync*)m_filters->m_model_template.get();
	row_sync->prep(x_strides(a_b));
	m_filters->prep(y_strides(a_a));
}

void cnl::unroll(size_t a_a, size_t a_b) {
	size_t x_filters = x_strides(a_b);
	size_t y_filters = y_strides(a_a);
	m_filters->unroll(y_filters);
	for (int i = 0; i < m_filters->m_unrolled.size(); i++) {
		sync* row_sync = (sync*)m_filters->m_unrolled[i].get();
		row_sync->unroll(x_filters);
	}
}

size_t cnl::x_strides(size_t a_width) {
	return (a_width - m_filter_width) / m_stride_len + 1;
}

size_t cnl::x_strides() {
	return (m_input_max_width - m_filter_width) / m_stride_len + 1;
}

size_t cnl::y_strides(size_t a_height) {
	return (a_height - m_filter_height) / m_stride_len + 1;
}

size_t cnl::y_strides() {
	return (m_input_max_height - m_filter_height) / m_stride_len + 1;
}