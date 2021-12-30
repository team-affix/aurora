#include "affix-base/pch.h"
#include "ntm_addresser.h"

using aurora::models::ntm_addresser;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::params::param_sgd;
using aurora::maths::tensor;
using std::vector;

ntm_addresser::~ntm_addresser() {

}

ntm_addresser::ntm_addresser() {

}

ntm_addresser::ntm_addresser(size_t a_memory_height, size_t a_memory_width, vector<int> a_valid_shifts) {
	m_memory_height = a_memory_height;
	m_memory_width = a_memory_width;
	m_shift_units = a_valid_shifts.size();
	m_internal_content_addresser = new ntm_content_addresser(a_memory_height, a_memory_width);
	m_internal_location_addresser = new ntm_location_addresser(a_memory_height, a_valid_shifts);
}

void ntm_addresser::param_recur(const function<void(Param&)>& a_func) {
	m_internal_content_addresser->param_recur(a_func);
	m_internal_location_addresser->param_recur(a_func);
}

model* ntm_addresser::clone(const function<Param(Param&)>& a_func) {
	ntm_addresser* result = new ntm_addresser();
	result->m_memory_height = m_memory_height;
	result->m_memory_width = m_memory_width;
	result->m_shift_units = m_shift_units;
	result->m_internal_content_addresser = (ntm_content_addresser*)m_internal_content_addresser->clone(a_func);
	result->m_internal_location_addresser = (ntm_location_addresser*)m_internal_location_addresser->clone(a_func);
	return result;
}

void ntm_addresser::fwd() {
	m_internal_content_addresser->fwd();
	m_internal_location_addresser->fwd();
}

void ntm_addresser::bwd() {
	m_internal_location_addresser->bwd();
	m_internal_content_addresser->bwd();
}

void ntm_addresser::model_recur(const function<void(model*)>& a_func) {
	a_func(this);
	m_internal_content_addresser->model_recur(a_func);
	m_internal_location_addresser->model_recur(a_func);
}

void ntm_addresser::compile() {
	m_x = tensor::new_2d(m_memory_height, m_memory_width);
	m_x_grad = tensor::new_2d(m_memory_height, m_memory_width);
	m_y = tensor::new_1d(m_memory_height);
	m_y_grad = tensor::new_1d(m_memory_height);
	m_key = tensor::new_1d(m_memory_width);
	m_key_grad = tensor::new_1d(m_memory_width);
	m_beta = tensor::new_1d(1);
	m_beta_grad = tensor::new_1d(1);
	m_wx = tensor::new_1d(m_memory_height);
	m_wx_grad = tensor::new_1d(m_memory_height);
	m_wy = tensor::new_1d(m_memory_height);
	m_wy_grad = tensor::new_1d(m_memory_height);
	m_g = tensor::new_1d(1);
	m_g_grad = tensor::new_1d(1);
	m_s = tensor::new_1d(m_shift_units);
	m_s_grad = tensor::new_1d(m_shift_units);
	m_gamma = tensor::new_1d(1);
	m_gamma_grad = tensor::new_1d(1);

	m_internal_content_addresser->compile();
	m_internal_location_addresser->compile();

	m_key.group_join(m_internal_content_addresser->m_key);
	m_key_grad.group_join(m_internal_content_addresser->m_key_grad);
	m_beta.group_join(m_internal_content_addresser->m_beta);
	m_beta_grad.group_join(m_internal_content_addresser->m_beta_grad);
	m_wx.group_join(m_internal_location_addresser->m_wx);
	m_wx_grad.group_join(m_internal_location_addresser->m_wx_grad);
	m_wy.group_join(m_internal_location_addresser->m_wy);
	m_wy_grad.group_join(m_internal_location_addresser->m_wy_grad);
	m_g.group_join(m_internal_location_addresser->m_g);
	m_g_grad.group_join(m_internal_location_addresser->m_g_grad);
	m_s.group_join(m_internal_location_addresser->m_s);
	m_s_grad.group_join(m_internal_location_addresser->m_s_grad);
	m_gamma.group_join(m_internal_location_addresser->m_gamma);
	m_gamma_grad.group_join(m_internal_location_addresser->m_gamma_grad);

	m_x.group_join(m_internal_content_addresser->m_x);
	m_x_grad.group_join(m_internal_content_addresser->m_x_grad);
	m_internal_content_addresser->m_y.group_join(m_internal_location_addresser->m_x);
	m_internal_content_addresser->m_y_grad.group_join(m_internal_location_addresser->m_x_grad);
	m_internal_location_addresser->m_y.group_join(m_y);
	m_internal_location_addresser->m_y_grad.group_join(m_y_grad);

}
