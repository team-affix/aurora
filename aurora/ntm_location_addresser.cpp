#include "affix-base/pch.h"
#include "ntm_location_addresser.h"

using aurora::models::ntm_location_addresser;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::params::param_sgd;
using aurora::maths::tensor;
using std::vector;

ntm_location_addresser::~ntm_location_addresser() {

}

ntm_location_addresser::ntm_location_addresser() {

}

ntm_location_addresser::ntm_location_addresser(size_t a_memory_height, vector<int> a_valid_shifts) {
	m_memory_height = a_memory_height;
	m_shift_units = a_valid_shifts.size();
	m_internal_interpolate = new interpolate(a_memory_height);
	m_internal_shift = new shift(a_memory_height, a_valid_shifts);
	m_internal_power = new power(a_memory_height);
	m_internal_normalize = new normalize(a_memory_height);
}

void ntm_location_addresser::param_recur(const function<void(Param&)>& a_func) {
	m_internal_interpolate->param_recur(a_func);
	m_internal_shift->param_recur(a_func);
	m_internal_power->param_recur(a_func);
	m_internal_normalize->param_recur(a_func);
}

model* ntm_location_addresser::clone(const function<Param(Param&)>& a_func) {
	ntm_location_addresser* result = new ntm_location_addresser();
	result->m_memory_height = m_memory_height;
	result->m_shift_units = m_shift_units;
	result->m_internal_interpolate = (interpolate*)m_internal_interpolate->clone(a_func);
	result->m_internal_shift = (shift*)m_internal_shift->clone(a_func);
	result->m_internal_power = (power*)m_internal_power->clone(a_func);
	result->m_internal_normalize = (normalize*)m_internal_normalize->clone(a_func);
	return result;
}

void ntm_location_addresser::fwd() {
	m_internal_interpolate->fwd();
	m_internal_shift->fwd();
	m_internal_power->fwd();
	m_internal_normalize->fwd();
}

void ntm_location_addresser::bwd() {
	m_y_grad.add_1d(m_wy_grad, m_internal_normalize->m_y_grad);
	m_internal_normalize->bwd();
	m_internal_power->bwd();
	m_internal_shift->bwd();
	m_internal_interpolate->bwd();
}

void ntm_location_addresser::signal(const tensor& a_y_des) {
	m_y.sub_1d(a_y_des, m_y_grad);
}

void ntm_location_addresser::model_recur(const function<void(model*)>& a_func) {
	a_func(this);
	m_internal_interpolate->model_recur(a_func);
	m_internal_shift->model_recur(a_func);
	m_internal_power->model_recur(a_func);
	m_internal_normalize->model_recur(a_func);
}

void ntm_location_addresser::compile() {
	m_x = tensor::new_1d(m_memory_height);
	m_x_grad = tensor::new_1d(m_memory_height);
	m_y = tensor::new_1d(m_memory_height);
	m_y_grad = tensor::new_1d(m_memory_height);
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

	m_internal_interpolate->compile();
	m_internal_shift->compile();
	m_internal_power->compile();
	m_internal_normalize->compile();
	
	m_g.group_join(m_internal_interpolate->m_amount);
	m_g_grad.group_join(m_internal_interpolate->m_amount_grad);
	m_s.group_join(m_internal_shift->m_amount);
	m_s_grad.group_join(m_internal_shift->m_amount_grad);
	m_gamma.group_join(m_internal_power->m_amount);
	m_gamma_grad.group_join(m_internal_power->m_amount_grad);

	m_wx.group_join(m_internal_interpolate->m_x[0]);
	m_wx[0].val() = 1; // MUST INITIALIZE wx TO BE A NORMALIZED DISTRIBUTION
	m_wx_grad.group_join(m_internal_interpolate->m_x_grad[0]);
	m_x.group_join(m_internal_interpolate->m_x[1]);
	m_x_grad.group_join(m_internal_interpolate->m_x_grad[1]);
	m_internal_interpolate->m_y.group_join(m_internal_shift->m_x);
	m_internal_interpolate->m_y_grad.group_join(m_internal_shift->m_x_grad);
	m_internal_shift->m_y.group_join(m_internal_power->m_x);
	m_internal_shift->m_y_grad.group_join(m_internal_power->m_x_grad);
	m_internal_power->m_y.group_join(m_internal_normalize->m_x);
	m_internal_power->m_y_grad.group_join(m_internal_normalize->m_x_grad);
	m_internal_normalize->m_y.group_join(m_y);
	m_y.group_join(m_wy);

}