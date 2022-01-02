#include "affix-base/pch.h"
#include "att_lstm_ts.h"

using aurora::models::att_lstm_ts;
using std::vector;
using aurora::maths::tensor;

att_lstm_ts::~att_lstm_ts() {

}

att_lstm_ts::att_lstm_ts() {

}

att_lstm_ts::att_lstm_ts(size_t a_units, vector<size_t> a_h_dims) {
	this->m_units = a_units;
	vector<size_t> l_dims;
	l_dims.push_back(2 * a_units);
	l_dims.insert(l_dims.end(), a_h_dims.begin(), a_h_dims.end());
	l_dims.push_back(1);
	// INITIALIZE NEURONS IN TNN
	vector<Model> neurons = vector<Model>(l_dims.size());
	for (int i = 0; i < l_dims.size() - 1; i++)
		neurons[i] = pseudo::nlr(0.3);
	neurons.push_back(pseudo::nsm());
	m_model_template = pseudo::tnn(l_dims, neurons);
	m_models = new sync(m_model_template);
}

void att_lstm_ts::param_recur(const std::function<void(aurora::params::Param&)>& a_func) {
	m_model_template->param_recur(a_func);
}

aurora::models::model* att_lstm_ts::clone(const std::function<aurora::params::Param(aurora::params::Param&)>& a_func) {
	att_lstm_ts* result = new att_lstm_ts();
	result->m_units = m_units;
	result->m_htx = m_htx.clone();
	result->m_htx_grad = m_htx_grad.clone();
	result->m_model_template = m_model_template->clone(a_func);
	result->m_models = (sync*)m_models->clone(a_func);
	return result;
}

void att_lstm_ts::fwd() {
	m_models->fwd();
	m_y.clear();
	for (int i = 0; i < m_models->m_unrolled.size(); i++) {
		double att_factor = m_models->m_y[i][0].val();
		for (int j = 0; j < m_units; j++)
			m_y[j].val() += m_x[i][j].val() * att_factor;
	}
}

void att_lstm_ts::bwd() {
	for (int i = 0; i < m_models->m_unrolled.size(); i++) {
		double att_factor = m_models->m_y[i][0].val();
		double att_factor_grad = 0;
		for (int j = 0; j < m_units; j++)
			att_factor_grad += m_y_grad[j].val() * m_x[i][j].val();
		m_models->m_unrolled[i]->m_y_grad[0].val() = att_factor_grad;
	}
	m_models->bwd();

	m_htx_grad.clear();
	for (int i = 0; i < m_models->m_unrolled.size(); i++) {
		double att_factor = m_models->m_y[i][0].val();
		for (int j = 0; j < m_units; j++)
			m_x_grad[i][j].val() += m_y_grad[j].val() * att_factor;

		// LOOP THROUGH HIDDEN STATE INPUT GRADIENT
		for (int j = 0; j < m_units; j++)
			m_htx_grad[j].val() += m_models->m_unrolled[i]->m_x_grad[j];
	}
}

void att_lstm_ts::model_recur(const std::function<void(model*)>& a_func) {
	m_model_template->model_recur(a_func);
}

void att_lstm_ts::compile() {
	m_models->compile();
	this->m_x = tensor::new_2d(m_models->m_prepared.size(), m_units);
	this->m_x_grad = tensor::new_2d(m_models->m_prepared.size(), m_units);
	this->m_y = tensor::new_1d(m_units);
	this->m_y_grad = tensor::new_1d(m_units);
	this->m_htx = tensor::new_1d(m_units);
	this->m_htx_grad = tensor::new_1d(m_units);
	for (int i = 0; i < m_models->m_prepared.size(); i++) {
		// cat ORDER: HT, **THEN** XT
		tensor htx_range = m_models->m_prepared[i]->m_x.range(0, m_units);
		tensor x_range = m_models->m_prepared[i]->m_x.range(m_units, m_units);
		tensor x_grad_range = m_models->m_prepared[i]->m_x_grad.range(m_units, m_units);
		htx_range.group_link(m_htx);
		x_range.group_link(m_x[i]);
		x_grad_range.group_link(m_x_grad[i]);
	}
}

void att_lstm_ts::prep(size_t a_n) {
	m_models->prep(a_n);
}

void att_lstm_ts::unroll(size_t a_n) {
	m_models->unroll(a_n);
}
