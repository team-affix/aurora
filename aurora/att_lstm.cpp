#include "affix-base/pch.h"
#include "att_lstm.h"

using aurora::models::att_lstm;
using aurora::maths::tensor;

att_lstm::~att_lstm() {

}

att_lstm::att_lstm() {

}

att_lstm::att_lstm(size_t a_units, std::vector<size_t> a_h_dims) {
	this->m_units = a_units;
	m_models = new sync(new att_lstm_ts(a_units, a_h_dims));
	m_internal_lstm = new lstm(a_units);
}

void att_lstm::param_recur(const std::function<void(aurora::params::Param&)>& a_func) {
	m_models->param_recur(a_func);
	m_internal_lstm->param_recur(a_func);
}

aurora::models::model* att_lstm::clone(const std::function<aurora::params::Param(aurora::params::Param&)>& a_func) {
	att_lstm* result = new att_lstm();
	result->m_units = m_units;
	result->m_models = (sync*)m_models->clone(a_func);
	result->m_internal_lstm = (lstm*)m_internal_lstm->clone(a_func);
	return result;
}

void att_lstm::fwd() {
	for (int i = 0; i < m_models->m_unrolled.size(); i++) {
		m_models->m_unrolled[i]->fwd();
		m_internal_lstm->m_unrolled[i]->fwd();
	}
}

void att_lstm::bwd() {
	m_x_grad.clear();
	for (int i = m_models->m_unrolled.size() - 1; i >= 0; i--) {
		att_lstm_ts* ats = (att_lstm_ts*)m_models->m_unrolled[i].get();
		lstm_ts* lts = m_internal_lstm->m_unrolled[i].get();
		lts->bwd();
		ats->bwd();
		lts->m_htx_grad.add_1d(ats->m_htx_grad, lts->m_htx_grad);
		m_x_grad.add_2d(ats->m_x_grad, m_x_grad);
	}
}

void att_lstm::model_recur(const std::function<void(model*)>& a_func) {
	m_models->model_recur(a_func);
	m_internal_lstm->model_recur(a_func);
}

void att_lstm::compile() {
	size_t l_a = m_models->m_prepared.size();
	size_t l_b = ((att_lstm_ts*)m_models->m_model_template.get())->m_models->m_prepared.size();
	this->m_x = tensor::new_2d(l_b, m_units);
	this->m_x_grad = tensor::new_2d(l_b, m_units);
	this->m_y = tensor::new_2d(l_a, m_units);
	this->m_y_grad = tensor::new_2d(l_a, m_units);
	m_models->compile();
	m_internal_lstm->compile();
	this->m_y.link(m_internal_lstm->m_y);
	this->m_y_grad.link(m_internal_lstm->m_y_grad);
	for (int i = 0; i < l_a; i++) {
		att_lstm_ts* ats = (att_lstm_ts*)m_models->m_prepared[i].get();
		lstm_ts* lts = m_internal_lstm->m_prepared[i].get();
		this->m_x.link(ats->m_x);
		ats->m_htx.link(lts->m_htx);
		lts->m_x.link(ats->m_y);
		lts->m_x_grad.link(ats->m_y_grad);
	}
}

void att_lstm::prep(size_t a_n, size_t b_n) {
	m_internal_lstm->prep(a_n);
	((att_lstm_ts*)m_models->m_model_template.get())->prep(b_n);
	m_models->prep(a_n);
}

void att_lstm::unroll(size_t a_n, size_t b_n) {
	m_internal_lstm->unroll(a_n);
	m_models->unroll(a_n);
	for (int i = 0; i < a_n; i++)
		((att_lstm_ts*)m_models->m_unrolled[i].get())->unroll(b_n);
}
