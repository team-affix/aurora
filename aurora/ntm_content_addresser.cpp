#include "affix-base/pch.h"
#include "ntm_content_addresser.h"

using aurora::models::ntm_content_addresser;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::params::param_sgd;
using aurora::maths::tensor;
using std::vector;

ntm_content_addresser::~ntm_content_addresser() {

}

ntm_content_addresser::ntm_content_addresser() {

}

ntm_content_addresser::ntm_content_addresser(size_t a_memory_height, size_t a_memory_width) {
	m_memory_height = a_memory_height;
	m_memory_width = a_memory_width;
	m_internal_similarity = new sync(new cos_sim(a_memory_width));
	m_internal_sparsify = new ntm_sparsify(m_memory_height);
	m_internal_normalize = new normalize(m_memory_height);
}

void ntm_content_addresser::param_recur(const function<void(Param&)>& a_func) {

}

model* ntm_content_addresser::clone(const function<Param(Param&)>& a_func) {
	ntm_content_addresser* result = new ntm_content_addresser();
	result->m_memory_height = m_memory_height;
	result->m_memory_width = m_memory_width;
	result->m_internal_similarity = (sync*)m_internal_similarity->clone(a_func);
	result->m_internal_sparsify = (ntm_sparsify*)m_internal_sparsify->clone(a_func);
	result->m_internal_normalize = (normalize*)m_internal_normalize->clone(a_func);
	return result;
}

void ntm_content_addresser::fwd() {
	m_internal_similarity->fwd();
	m_internal_sparsify->fwd();
	m_internal_normalize->fwd();
}

void ntm_content_addresser::bwd() {
	m_internal_normalize->bwd();
	m_internal_sparsify->bwd();
	m_internal_similarity->bwd();

	m_key_grad.clear();
	for (int i = 0; i < m_memory_height; i++)
		m_key_grad.add_1d(m_internal_similarity->m_x_grad[i][0], m_key_grad);
}

void ntm_content_addresser::model_recur(const function<void(model*)>& a_func) {
	a_func(this);
	m_internal_similarity->model_recur(a_func);
	m_internal_sparsify->model_recur(a_func);
	m_internal_normalize->model_recur(a_func);
}

void ntm_content_addresser::compile() {
	m_x = tensor::new_2d(m_memory_height, m_memory_width);
	m_x_grad = tensor::new_2d(m_memory_height, m_memory_width);
	m_y = tensor::new_1d(m_memory_height);
	m_y_grad = tensor::new_1d(m_memory_height);
	m_key = tensor::new_1d(m_memory_width);
	m_key_grad = tensor::new_1d(m_memory_width);
	m_beta = tensor::new_1d(1);
	m_beta_grad = tensor::new_1d(1);

	m_internal_similarity->prep(m_memory_height);
	m_internal_similarity->compile();
	m_internal_similarity->unroll(m_memory_height);

	m_internal_sparsify->compile();
	m_internal_normalize->compile();

	for (int i = 0; i < m_memory_height; i++) {
		m_key.group_join(m_internal_similarity->m_x[i][0]);
		m_x[i].group_join(m_internal_similarity->m_x[i][1]);
		m_x_grad[i].group_join(m_internal_similarity->m_x_grad[i][1]);
	}

	m_internal_similarity->m_y.group_join(m_internal_sparsify->m_x);
	m_internal_similarity->m_y_grad.group_join(m_internal_sparsify->m_x_grad);

	m_internal_sparsify->m_y.group_join(m_internal_normalize->m_x);
	m_internal_sparsify->m_y_grad.group_join(m_internal_normalize->m_x_grad);

	m_internal_normalize->m_y.group_join(m_y);
	m_internal_normalize->m_y_grad.group_join(m_y_grad);

	m_beta.group_join(m_internal_sparsify->m_beta);
	m_beta_grad.group_join(m_internal_sparsify->m_beta_grad);

}