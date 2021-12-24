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
	this->units = a_units;
	this->forget_gate = new layer(units, pseudo::nsm());
	this->limit_gate = new layer(units, pseudo::nsm());
	this->input_gate = new layer(units, pseudo::nth());
	this->output_gate = new layer(units, pseudo::nsm());
	this->tanh_gate = new layer(units, pseudo::nth());
}

lstm_ts::lstm_ts(size_t a_units, Layer a_forget_gate, Layer a_limit_gate, Layer a_input_gate, Layer a_output_gate, Layer a_tanh_gate) {
	this->units = a_units;
	this->forget_gate = a_forget_gate;
	this->limit_gate = a_limit_gate;
	this->input_gate = a_input_gate;
	this->output_gate = a_output_gate;
	this->tanh_gate = a_tanh_gate;
}

void lstm_ts::param_recur(function<void(Param&)> a_func) {
	forget_gate->param_recur(a_func);
	limit_gate->param_recur(a_func);
	input_gate->param_recur(a_func);
	output_gate->param_recur(a_func);
	tanh_gate->param_recur(a_func);
}

model* lstm_ts::clone(function<Param(Param&)> a_func) {
	return new lstm_ts(units, (layer*)forget_gate->clone(a_func), (layer*)limit_gate->clone(a_func), (layer*)input_gate->clone(a_func), (layer*)output_gate->clone(a_func), (layer*)tanh_gate->clone(a_func));
}

void lstm_ts::fwd() {
	x.add_1d(htx, gate_x);
	forget_gate->fwd();
	limit_gate->fwd();
	input_gate->fwd();
	output_gate->fwd();

	forget_gate->y.mul_1d(ctx, comp_0);
	limit_gate->y.mul_1d(input_gate->y, comp_1);
	comp_0.add_1d(comp_1, cty);
	tanh_gate->fwd();
	tanh_gate->y.mul_1d(output_gate->y, hty);
}

void lstm_ts::bwd() {
	y_grad.add_1d(hty_grad, comp_0);
	comp_0.mul_1d(tanh_gate->y, output_gate->y_grad);
	comp_0.mul_1d(output_gate->y, tanh_gate->y_grad);
	
	tanh_gate->bwd();

	tanh_gate->x_grad.add_1d(cty_grad, comp_0);
	comp_0.mul_1d(input_gate->y, limit_gate->y_grad);
	comp_0.mul_1d(limit_gate->y, input_gate->y_grad);
	comp_0.mul_1d(ctx, forget_gate->y_grad);
	comp_0.mul_1d(forget_gate->y, ctx_grad);

	output_gate->bwd();
	input_gate->bwd();
	limit_gate->bwd();
	forget_gate->bwd();

	forget_gate->x_grad.add_1d(limit_gate->x_grad, comp_0);
	input_gate->x_grad.add_1d(output_gate->x_grad, comp_1);
	comp_0.add_1d(comp_1, htx_grad);
}

void lstm_ts::signal(const tensor& a_y_des) {
	y.sub_1d(a_y_des, y_grad);
}

void lstm_ts::model_recur(function<void(model*)> a_func) {
	a_func(this);
	forget_gate->model_recur(a_func);
	limit_gate->model_recur(a_func);
	input_gate->model_recur(a_func);
	output_gate->model_recur(a_func);
	tanh_gate->model_recur(a_func);
}

void lstm_ts::compile() {
	this->x = tensor::new_1d(units);
	this->y = tensor::new_1d(units);
	this->x_grad = tensor::new_1d(units);
	this->y_grad = tensor::new_1d(units);
	this->ctx = tensor::new_1d(units);
	this->cty = tensor::new_1d(units);
	this->htx = tensor::new_1d(units);
	this->hty = tensor::new_1d(units);
	this->ctx_grad = tensor::new_1d(units);
	this->cty_grad = tensor::new_1d(units);
	this->htx_grad = tensor::new_1d(units);
	this->hty_grad = tensor::new_1d(units);
	this->gate_x = tensor::new_1d(units);
	this->comp_0 = tensor::new_1d(units);
	this->comp_1 = tensor::new_1d(units);
	forget_gate->compile();
	limit_gate->compile();
	input_gate->compile();
	output_gate->compile();
	tanh_gate->compile();
	gate_x.group_add(forget_gate->x);
	gate_x.group_add(limit_gate->x);
	gate_x.group_add(input_gate->x);
	gate_x.group_add(output_gate->x);
	cty.group_add(tanh_gate->x);
	hty.group_add(y);
	htx_grad.group_add(x_grad);
}
