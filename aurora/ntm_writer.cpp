#include "affix-base/pch.h"
#include "ntm_writer.h"
#include <iostream>

using aurora::models::ntm_writer;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::params::param_sgd;
using aurora::maths::tensor;
using std::vector;

ntm_writer::~ntm_writer() {

}

ntm_writer::ntm_writer() {

}

ntm_writer::ntm_writer(size_t a_memory_height, size_t a_memory_width, vector<int> a_valid_shifts, vector<size_t> a_head_hidden_dims) {
	m_memory_height = a_memory_height;
	m_memory_width = a_memory_width;
	m_internal_head = new ntm_wh(a_memory_width, a_head_hidden_dims, a_valid_shifts.size());
	m_internal_addresser = new ntm_addresser(a_memory_height, a_memory_width, a_valid_shifts);
}

void ntm_writer::param_recur(const function<void(Param&)>& a_func) {
	m_internal_head->param_recur(a_func);
	m_internal_addresser->param_recur(a_func);
}

model* ntm_writer::clone(const function<Param(Param&)>& a_func) {
	ntm_writer* result = new ntm_writer();
	result->m_memory_height = m_memory_height;
	result->m_memory_width = m_memory_width;
	result->m_internal_head = (ntm_wh*)m_internal_head->clone(a_func);
	result->m_internal_addresser = (ntm_addresser*)m_internal_addresser->clone(a_func);
	return result;
}

void ntm_writer::fwd() {
	m_internal_head->fwd();
	m_internal_addresser->fwd();
	for (int i = 0; i < m_memory_height; i++)
		for (int j = 0; j < m_memory_width; j++) {
			m_weighted_erase_compliment[i][j].val() = 1.0 - m_internal_addresser->m_y[i] * m_internal_head->m_e[j];
			m_y[i][j].val() =
				m_mx[i][j] * m_weighted_erase_compliment[i][j];
			m_y[i][j].val() +=
				m_internal_addresser->m_y[i] *
				m_internal_head->m_a[j];
		}
}

void ntm_writer::bwd() {
	m_internal_addresser->m_y_grad.clear();
	m_internal_head->m_e_grad.clear();
	m_internal_head->m_a_grad.clear();
	for (int i = 0; i < m_memory_height; i++)
		for (int j = 0; j < m_memory_width; j++) {
			m_mx_grad[i][j].val() = m_y_grad[i][j] * m_weighted_erase_compliment[i][j];
			m_internal_addresser->m_y_grad[i].val() += m_y_grad[i][j] * 
				(-m_mx[i][j] * m_internal_head->m_e[j] + m_internal_head->m_a[j]);
			m_internal_head->m_e_grad[j].val() += m_y_grad[i][j] *
				-m_mx[i][j] * m_internal_addresser->m_y[i];
			m_internal_head->m_a_grad[j].val() += m_y_grad[i][j] *
				m_internal_addresser->m_y[i];
		}
	m_internal_addresser->bwd();
	m_mx_grad.add_2d(m_internal_addresser->m_x_grad, m_mx_grad);
	m_internal_head->bwd();
}

void ntm_writer::model_recur(const function<void(model*)>& a_func) {
	a_func(this);
	m_internal_head->model_recur(a_func);
	m_internal_addresser->model_recur(a_func);
}

void ntm_writer::compile() {
	m_x = tensor::new_1d(m_memory_width);
	m_x_grad = tensor::new_1d(m_memory_width);
	m_y = tensor::new_2d(m_memory_height, m_memory_width);
	m_y_grad = tensor::new_2d(m_memory_height, m_memory_width);
	m_weighted_erase_compliment = tensor::new_2d(m_memory_height, m_memory_width);
	m_mx = tensor::new_2d(m_memory_height, m_memory_width);
	m_mx_grad = tensor::new_2d(m_memory_height, m_memory_width);
	m_wx = tensor::new_1d(m_memory_height);
	m_wx_grad = tensor::new_1d(m_memory_height);
	m_wy = tensor::new_1d(m_memory_height);
	m_wy_grad = tensor::new_1d(m_memory_height);

	m_internal_head->compile();
	m_internal_addresser->compile();

	m_internal_head->m_internal_rh->m_key.link(m_internal_addresser->m_key);
	m_internal_head->m_internal_rh->m_key_grad.link(m_internal_addresser->m_key_grad);
	m_internal_head->m_internal_rh->m_beta.link(m_internal_addresser->m_beta);
	m_internal_head->m_internal_rh->m_beta_grad.link(m_internal_addresser->m_beta_grad);
	m_internal_head->m_internal_rh->m_g.link(m_internal_addresser->m_g);
	m_internal_head->m_internal_rh->m_g_grad.link(m_internal_addresser->m_g_grad);
	m_internal_head->m_internal_rh->m_s.link(m_internal_addresser->m_s);
	m_internal_head->m_internal_rh->m_s_grad.link(m_internal_addresser->m_s_grad);
	m_internal_head->m_internal_rh->m_gamma.link(m_internal_addresser->m_gamma);
	m_internal_head->m_internal_rh->m_gamma_grad.link(m_internal_addresser->m_gamma_grad);

	m_x.link(m_internal_head->m_x);
	m_x_grad.link(m_internal_head->m_x_grad);
	m_mx.link(m_internal_addresser->m_x);
	m_wx.link(m_internal_addresser->m_wx);
	m_wx_grad.link(m_internal_addresser->m_wx_grad);
	m_internal_addresser->m_wy.link(m_wy);
	m_internal_addresser->m_wy_grad.link(m_wy_grad);
}
