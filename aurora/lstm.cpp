#include "affix-base/pch.h"
#include "lstm.h"

using aurora::models::lstm;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::params::param_sgd;
using aurora::maths::tensor;
using std::vector;

lstm::~lstm() {

}

lstm::lstm() {

}

lstm::lstm(size_t a_units) {
	this->m_units = a_units;
	m_lstm_ts_template = new lstm_ts(m_units);
}

void lstm::param_recur(const function<void(Param&)>& a_func) {
	m_lstm_ts_template->param_recur(a_func);
}

model* lstm::clone(const function<Param(Param&)>& a_func) {
	lstm* result = new lstm();
	result->m_units = m_units;
	result->m_lstm_ts_template = (lstm_ts*)m_lstm_ts_template->clone(a_func);
	result->prep(m_prepared.size());
	result->unroll(m_unrolled.size());
	return result;
}

void lstm::fwd() {
	for (int i = 0; i < m_unrolled.size(); i++)
		m_unrolled[i]->fwd();
}

void lstm::bwd() {
	for (int i = m_unrolled.size() - 1; i >= 0; i--)
		m_unrolled[i]->bwd();
}

void lstm::model_recur(const function<void(model*)>& a_func) {
	a_func(this);
	m_lstm_ts_template->model_recur(a_func);
}

void lstm::compile() {
	this->m_x = tensor::new_2d(m_prepared.size(), m_units);
	this->m_y = tensor::new_2d(m_prepared.size(), m_units);
	this->m_x_grad = tensor::new_2d(m_prepared.size(), m_units);
	this->m_y_grad = tensor::new_2d(m_prepared.size(), m_units);
	this->m_ctx = tensor::new_1d(m_units);
	this->m_cty = tensor::new_1d(m_units);
	this->m_ctx_grad = tensor::new_1d(m_units);
	this->m_cty_grad = tensor::new_1d(m_units);
	this->m_htx = tensor::new_1d(m_units);
	this->m_hty = tensor::new_1d(m_units);
	this->m_htx_grad = tensor::new_1d(m_units);
	this->m_hty_grad = tensor::new_1d(m_units);
	tensor* ct = &m_ctx;
	tensor* ht = &m_htx;
	tensor* ct_grad = &m_ctx_grad;
	tensor* ht_grad = &m_htx_grad;
	for (int i = 0; i < m_prepared.size(); i++) {
		m_prepared[i]->compile();
		m_x[i].group_link(m_prepared[i]->m_x);
		m_y[i].group_link(m_prepared[i]->m_y);
		m_x_grad[i].group_link(m_prepared[i]->m_x_grad);
		m_y_grad[i].group_link(m_prepared[i]->m_y_grad);
		m_prepared[i]->m_ctx.group_link(*ct);
		m_prepared[i]->m_htx.group_link(*ht);
		m_prepared[i]->m_ctx_grad.group_link(*ct_grad);
		m_prepared[i]->m_htx_grad.group_link(*ht_grad);
		ct = &m_prepared[i]->m_cty;
		ht = &m_prepared[i]->m_hty;
		ct_grad = &m_prepared[i]->m_cty_grad;
		ht_grad = &m_prepared[i]->m_hty_grad;
	}
	m_cty.group_link(*ct);
	m_hty.group_link(*ht);
	m_cty_grad.group_link(*ct_grad);
	m_hty_grad.group_link(*ht_grad);
}

void lstm::prep(size_t a_n) {
	m_prepared.clear();
	m_prepared.resize(a_n);
	for (int i = 0; i < a_n; i++)
		m_prepared.at(i) = (lstm_ts*)m_lstm_ts_template->clone();
}

void lstm::unroll(size_t a_n) {
	m_unrolled.clear();
	m_unrolled.resize(a_n);
	for (int i = 0; i < a_n; i++)
		m_unrolled.at(i) = m_prepared.at(i);
}
