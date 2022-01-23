#include "affix-base/pch.h"
#include "ntm.h"

using aurora::models::ntm;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::params::param_sgd;
using aurora::maths::tensor;
using std::vector;

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
	vector<size_t> a_head_hidden_dims) {
	m_memory_height = a_memory_height;
	m_memory_width = a_memory_width;
	
	m_internal_lstm = new lstm(a_memory_width);

	m_ntm_ts_template = new ntm_ts(
		a_memory_height,
		a_memory_width,
		a_num_readers,
		a_num_writers,
		a_valid_shifts,
		a_head_hidden_dims);

}

void ntm::param_recur(const function<void(Param&)>& a_func) {
	m_internal_lstm->param_recur(a_func);
	m_ntm_ts_template->param_recur(a_func);
}

model* ntm::clone(const function<Param(Param&)>& a_func) {
	ntm* result = new ntm();
	result->m_memory_height = m_memory_height;
	result->m_memory_width = m_memory_width;
	result->m_internal_lstm = (lstm*)m_internal_lstm->clone(a_func);
	result->m_ntm_ts_template = (ntm_ts*)m_ntm_ts_template->clone(a_func);
	result->prep(m_prepared.size());
	result->unroll(m_prepared.size());
	return result;
}

void ntm::fwd() {
	for (int i = 0; i < m_unrolled.size(); i++) {
		m_internal_lstm->m_unrolled[i]->fwd();
		m_unrolled[i]->fwd();
		m_internal_lstm->m_unrolled[i]->m_hty.add_1d(m_unrolled[i]->m_y, m_internal_lstm->m_unrolled[i]->m_hty);
	}
}

void ntm::bwd() {
	for (int i = m_unrolled.size() - 1; i >= 0; i--) {
		m_unrolled[i]->bwd();
		m_internal_lstm->m_unrolled[i]->bwd();
	}
}

void ntm::model_recur(const function<void(model*)>& a_func) {
	a_func(this);
	m_internal_lstm->model_recur(a_func);
	m_ntm_ts_template->model_recur(a_func);
}

void ntm::compile() {
	m_x = tensor::new_2d(m_prepared.size(), m_memory_width);
	m_x_grad = tensor::new_2d(m_prepared.size(), m_memory_width);
	m_y = tensor::new_2d(m_prepared.size(), m_memory_width);
	m_y_grad = tensor::new_2d(m_prepared.size(), m_memory_width);
	m_mx = tensor::new_2d(m_memory_height, m_memory_width, 1.0);	
	m_mx_grad = tensor::new_2d(m_memory_height, m_memory_width);
	m_my = tensor::new_2d(m_memory_height, m_memory_width);
	m_my_grad = tensor::new_2d(m_memory_height, m_memory_width);

	// CREATE TENSORS FOR OUTERMOST WEIGHTINGS (wx,wy)
	m_read_wx = tensor::new_2d(m_ntm_ts_template->m_internal_readers.size(), m_memory_height);
	m_read_wx_grad = tensor::new_2d(m_ntm_ts_template->m_internal_readers.size(), m_memory_height);
	m_read_wy = tensor::new_2d(m_ntm_ts_template->m_internal_readers.size(), m_memory_height);
	m_read_wy_grad = tensor::new_2d(m_ntm_ts_template->m_internal_readers.size(), m_memory_height);
	m_write_wx = tensor::new_2d(m_ntm_ts_template->m_internal_writers.size(), m_memory_height);
	m_write_wx_grad = tensor::new_2d(m_ntm_ts_template->m_internal_writers.size(), m_memory_height);
	m_write_wy = tensor::new_2d(m_ntm_ts_template->m_internal_writers.size(), m_memory_height);
	m_write_wy_grad = tensor::new_2d(m_ntm_ts_template->m_internal_writers.size(), m_memory_height);

	m_internal_lstm->compile();
	m_x.group_link(m_internal_lstm->m_x);
	m_x_grad.group_link(m_internal_lstm->m_x_grad);

	tensor* l_mx = &m_mx;
	tensor* l_mx_grad = &m_mx_grad;

	tensor* l_read_wx = &m_read_wx;
	tensor* l_read_wx_grad = &m_read_wx_grad;
	tensor* l_write_wx = &m_write_wx;
	tensor* l_write_wx_grad = &m_write_wx_grad;

	for (int i = 0; i < m_prepared.size(); i++) {
		
		m_prepared[i]->compile();
		m_internal_lstm->m_y[i].group_link(m_prepared[i]->m_x);
		m_internal_lstm->m_y_grad[i].group_link(m_prepared[i]->m_x_grad);
		m_prepared[i]->m_y.group_link(m_y[i]);
		m_prepared[i]->m_y_grad.group_link(m_y_grad[i]);
		m_prepared[i]->m_hty_grad.group_link(m_internal_lstm->m_prepared[i]->m_hty_grad);
		
		m_prepared[i]->m_mx.group_link(*l_mx);
		l_mx_grad->group_link(m_prepared[i]->m_mx_grad);

		l_mx = &m_prepared[i]->m_my;
		l_mx_grad = &m_prepared[i]->m_my_grad;

		l_read_wx->group_link(m_prepared[i]->m_read_wx);
		l_read_wx_grad->group_link(m_prepared[i]->m_read_wx_grad);
		l_write_wx->group_link(m_prepared[i]->m_write_wx);
		l_write_wx_grad->group_link(m_prepared[i]->m_write_wx_grad);

		l_read_wx = &m_prepared[i]->m_read_wy;
		l_read_wx_grad = &m_prepared[i]->m_read_wy_grad;
		l_write_wx = &m_prepared[i]->m_write_wy;
		l_write_wx_grad = &m_prepared[i]->m_write_wy_grad;

	}

	l_mx->group_link(m_my);
	l_mx_grad->group_link(m_my_grad);

	l_read_wx->group_link(m_read_wy);
	l_read_wx_grad->group_link(m_read_wy_grad);
	l_write_wx->group_link(m_write_wy);
	l_write_wx_grad->group_link(m_write_wy_grad);

}

void ntm::prep(size_t a_n) {
	m_internal_lstm->prep(a_n);
	m_prepared.clear();
	m_prepared.resize(a_n);
	for (int i = 0; i < a_n; i++)
		m_prepared.at(i) = (ntm_ts*)m_ntm_ts_template->clone();
}

void ntm::unroll(size_t a_n) {
	m_internal_lstm->unroll(a_n);
	m_unrolled.clear();
	m_unrolled.resize(a_n);
	for (int i = 0; i < a_n; i++)
		m_unrolled.at(i) = m_prepared.at(i);
}
