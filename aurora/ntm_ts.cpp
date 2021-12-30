#include "affix-base/pch.h"
#include "ntm_ts.h"

using aurora::models::ntm_ts;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::params::param_sgd;
using aurora::maths::tensor;
using std::vector;

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
	vector<size_t> a_head_hidden_dims) {
	m_memory_height = a_memory_height;
	m_memory_width = a_memory_width;

	for (int i = 0; i < a_num_readers; i++)
		m_internal_readers.push_back(new ntm_reader(a_memory_height, a_memory_width, a_valid_shifts, a_head_hidden_dims));
	for (int i = 0; i < a_num_writers; i++)
		m_internal_writers.push_back(new ntm_writer(a_memory_height, a_memory_width, a_valid_shifts, a_head_hidden_dims));

}

void ntm_ts::param_recur(const function<void(Param&)>& a_func) {
	for (int i = 0; i < m_internal_readers.size(); i++)
		m_internal_readers[i]->param_recur(a_func);
	for (int i = 0; i < m_internal_writers.size(); i++)
		m_internal_writers[i]->param_recur(a_func);
}

model* ntm_ts::clone(const function<Param(Param&)>& a_func) {
	ntm_ts* result = new ntm_ts();
	result->m_memory_height = m_memory_height;
	result->m_memory_width = m_memory_width;
	for (int i = 0; i < m_internal_readers.size(); i++)
		result->m_internal_readers.push_back((ntm_reader*)m_internal_readers[i]->clone(a_func));
	for (int i = 0; i < m_internal_writers.size(); i++)
		result->m_internal_writers.push_back((ntm_writer*)m_internal_writers[i]->clone(a_func));
	return result;
}

void ntm_ts::fwd() {
	for (int i = 0; i < m_internal_writers.size(); i++)
		m_internal_writers[i]->fwd();
	m_y.clear();
	for (int i = 0; i < m_internal_readers.size(); i++) {
		m_internal_readers[i]->fwd();
		m_y.add_1d(m_internal_readers[i]->m_y, m_y);
	}
}

void ntm_ts::bwd() {
	m_x_grad.clear();
	m_accum_my_grad.pop(m_my_grad);
	m_y_grad.add_1d(m_hty_grad, m_reader_y_grad);
	for (int i = 0; i < m_internal_readers.size(); i++) {
		m_internal_readers[i]->bwd();
		m_x_grad.add_1d(m_internal_readers[i]->m_x_grad, m_x_grad);
		m_accum_my_grad.add_2d(m_internal_readers[i]->m_mx_grad, m_accum_my_grad);
	}
	for (int i = 0; i < m_internal_writers.size(); i++) {
		m_internal_writers[i]->bwd();
		m_x_grad.add_1d(m_internal_writers[i]->m_x_grad, m_x_grad);
	}
}

void ntm_ts::model_recur(const function<void(model*)>& a_func) {
	a_func(this);
	for (int i = 0; i < m_internal_readers.size(); i++)
		m_internal_readers[i]->model_recur(a_func);
	for (int i = 0; i < m_internal_writers.size(); i++)
		m_internal_writers[i]->model_recur(a_func);
}

void ntm_ts::compile() {
	m_x = tensor::new_1d(m_memory_width);
	m_x_grad = tensor::new_1d(m_memory_width);
	m_y = tensor::new_1d(m_memory_width);
	m_y_grad = tensor::new_1d(m_memory_width);
	m_mx = tensor::new_2d(m_memory_height, m_memory_width);
	m_mx_grad = tensor::new_2d(m_memory_height, m_memory_width);
	m_my = tensor::new_2d(m_memory_height, m_memory_width);
	m_my_grad = tensor::new_2d(m_memory_height, m_memory_width);
	m_reader_y_grad = tensor::new_1d(m_memory_width);
	m_accum_my_grad = tensor::new_2d(m_memory_height, m_memory_width);

	// CREATE TENSORS FOR OUTERMOST WEIGHTINGS (wx,wy)
	m_read_wx = tensor::new_2d(m_internal_readers.size(), m_memory_height);
	m_read_wx_grad = tensor::new_2d(m_internal_readers.size(), m_memory_height);
	m_read_wy = tensor::new_2d(m_internal_readers.size(), m_memory_height);
	m_read_wy_grad = tensor::new_2d(m_internal_readers.size(), m_memory_height);
	m_write_wx = tensor::new_2d(m_internal_writers.size(), m_memory_height);
	m_write_wx_grad = tensor::new_2d(m_internal_writers.size(), m_memory_height);
	m_write_wy = tensor::new_2d(m_internal_writers.size(), m_memory_height);
	m_write_wy_grad = tensor::new_2d(m_internal_writers.size(), m_memory_height);

	tensor* l_mx = &m_mx;
	tensor* l_mx_grad = &m_mx_grad;

	for (int i = 0; i < m_internal_writers.size(); i++) {
		m_internal_writers[i]->compile();
		m_internal_writers[i]->m_x.group_join_all_ranks(m_x);
		m_internal_writers[i]->m_mx.group_join_all_ranks(*l_mx);
		m_internal_writers[i]->m_mx_grad.group_join_all_ranks(*l_mx_grad);
		l_mx = &m_internal_writers[i]->m_y;
		l_mx_grad = &m_internal_writers[i]->m_y_grad;

		m_write_wx[i].group_join(m_internal_writers[i]->m_wx);
		m_write_wx_grad[i].group_join(m_internal_writers[i]->m_wx_grad);
		m_write_wy[i].group_join(m_internal_writers[i]->m_wy);
		m_write_wy_grad[i].group_join(m_internal_writers[i]->m_wy_grad);
	}

	l_mx->group_join_all_ranks(m_my);
	l_mx_grad->group_join_all_ranks(m_accum_my_grad);

	for (int i = 0; i < m_internal_readers.size(); i++) {
		m_internal_readers[i]->compile();
		m_internal_readers[i]->m_x.group_join_all_ranks(m_x);
		m_internal_readers[i]->m_mx.group_join_all_ranks(m_my);
		m_internal_readers[i]->m_y_grad.group_join_all_ranks(m_reader_y_grad);

		m_read_wx[i].group_join(m_internal_readers[i]->m_wx);
		m_read_wx_grad[i].group_join(m_internal_readers[i]->m_wx_grad);
		m_read_wy[i].group_join(m_internal_readers[i]->m_wy);
		m_read_wy_grad[i].group_join(m_internal_readers[i]->m_wy_grad);
	}

}