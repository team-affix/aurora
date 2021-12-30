#include "affix-base/pch.h"
#include "lstm_ts.h"
#include "pseudo_tnn.h"
#include "neuron.h"

using aurora::models::lstm_ts;
using aurora::models::layer;
using namespace aurora;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::params::param_sgd;
using aurora::maths::tensor;
using std::vector;

lstm_ts::~lstm_ts() {

}

lstm_ts::lstm_ts() {

}

lstm_ts::lstm_ts(size_t a_units) {
	this->m_units = a_units;
	this->m_forget_gate = new layer(m_units, pseudo::nsm());
	this->m_limit_gate = new layer(m_units, pseudo::nsm());
	this->m_input_gate = new layer(m_units, pseudo::nth());
	this->m_output_gate = new layer(m_units, pseudo::nsm());
	this->m_tanh_gate = new layer(m_units, pseudo::nth());
}

lstm_ts::lstm_ts(size_t a_units, Layer a_forget_gate, Layer a_limit_gate, Layer a_input_gate, Layer a_output_gate, Layer a_tanh_gate) {
	this->m_units = a_units;
	this->m_forget_gate = a_forget_gate;
	this->m_limit_gate = a_limit_gate;
	this->m_input_gate = a_input_gate;
	this->m_output_gate = a_output_gate;
	this->m_tanh_gate = a_tanh_gate;
}

void lstm_ts::param_recur(const function<void(Param&)>& a_func) {
	m_forget_gate->param_recur(a_func);
	m_limit_gate->param_recur(a_func);
	m_input_gate->param_recur(a_func);
	m_output_gate->param_recur(a_func);
	m_tanh_gate->param_recur(a_func);
}

model* lstm_ts::clone(const function<Param(Param&)>& a_func) {
	return new lstm_ts(m_units, (layer*)m_forget_gate->clone(a_func), (layer*)m_limit_gate->clone(a_func), (layer*)m_input_gate->clone(a_func), (layer*)m_output_gate->clone(a_func), (layer*)m_tanh_gate->clone(a_func));
}

void lstm_ts::fwd() {
	m_x.add_1d(m_htx, m_gate_x);
	m_forget_gate->fwd();
	m_limit_gate->fwd();
	m_input_gate->fwd();
	m_output_gate->fwd();

	m_forget_gate->m_y.mul_1d(m_ctx, m_comp_0);
	m_limit_gate->m_y.mul_1d(m_input_gate->m_y, m_comp_1);
	m_comp_0.add_1d(m_comp_1, m_cty);
	m_tanh_gate->fwd();
	m_tanh_gate->m_y.mul_1d(m_output_gate->m_y, m_hty);
}

void lstm_ts::bwd() {
	m_y_grad.add_1d(m_hty_grad, m_comp_0);
	m_comp_0.mul_1d(m_tanh_gate->m_y, m_output_gate->m_y_grad);
	m_comp_0.mul_1d(m_output_gate->m_y, m_tanh_gate->m_y_grad);
	
	m_tanh_gate->bwd();

	m_tanh_gate->m_x_grad.add_1d(m_cty_grad, m_comp_0);
	m_comp_0.mul_1d(m_input_gate->m_y, m_limit_gate->m_y_grad);
	m_comp_0.mul_1d(m_limit_gate->m_y, m_input_gate->m_y_grad);
	m_comp_0.mul_1d(m_ctx, m_forget_gate->m_y_grad);
	m_comp_0.mul_1d(m_forget_gate->m_y, m_ctx_grad);

	m_output_gate->bwd();
	m_input_gate->bwd();
	m_limit_gate->bwd();
	m_forget_gate->bwd();

	m_forget_gate->m_x_grad.add_1d(m_limit_gate->m_x_grad, m_comp_0);
	m_input_gate->m_x_grad.add_1d(m_output_gate->m_x_grad, m_comp_1);
	m_comp_0.add_1d(m_comp_1, m_htx_grad);
}

void lstm_ts::model_recur(const function<void(model*)>& a_func) {
	a_func(this);
	m_forget_gate->model_recur(a_func);
	m_limit_gate->model_recur(a_func);
	m_input_gate->model_recur(a_func);
	m_output_gate->model_recur(a_func);
	m_tanh_gate->model_recur(a_func);
}

void lstm_ts::compile() {
	this->m_x = tensor::new_1d(m_units);
	this->m_y = tensor::new_1d(m_units);
	this->m_x_grad = tensor::new_1d(m_units);
	this->m_y_grad = tensor::new_1d(m_units);
	this->m_ctx = tensor::new_1d(m_units);
	this->m_cty = tensor::new_1d(m_units);
	this->m_htx = tensor::new_1d(m_units);
	this->m_hty = tensor::new_1d(m_units);
	this->m_ctx_grad = tensor::new_1d(m_units);
	this->m_cty_grad = tensor::new_1d(m_units);
	this->m_htx_grad = tensor::new_1d(m_units);
	this->m_hty_grad = tensor::new_1d(m_units);
	this->m_gate_x = tensor::new_1d(m_units);
	this->m_comp_0 = tensor::new_1d(m_units);
	this->m_comp_1 = tensor::new_1d(m_units);
	m_forget_gate->compile();
	m_limit_gate->compile();
	m_input_gate->compile();
	m_output_gate->compile();
	m_tanh_gate->compile();
	m_gate_x.group_add(m_forget_gate->m_x);
	m_gate_x.group_add(m_limit_gate->m_x);
	m_gate_x.group_add(m_input_gate->m_x);
	m_gate_x.group_add(m_output_gate->m_x);
	m_cty.group_add(m_tanh_gate->m_x);
	m_hty.group_add(m_y);
	m_htx_grad.group_add(m_x_grad);
}
