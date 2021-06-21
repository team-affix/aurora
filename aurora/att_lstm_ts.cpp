#include "pch.h"
#include "att_lstm_ts.h"

using aurora::models::att_lstm_ts;

att_lstm_ts::~att_lstm_ts() {

}

att_lstm_ts::att_lstm_ts() {

}

att_lstm_ts::att_lstm_ts(size_t a_units, vector<size_t> a_h_dims) {
	this->units = a_units;
	vector<size_t> l_dims;
	l_dims.push_back(2 * a_units);
	l_dims.insert(l_dims.end(), a_h_dims.begin(), a_h_dims.end());
	l_dims.push_back(1);
	// INITIALIZE NEURONS IN TNN
	vector<Model> neurons = vector<Model>(l_dims.size());
	for (int i = 0; i < l_dims.size() - 1; i++)
		neurons[i] = pseudo::nlr(0.3);
	neurons.push_back(pseudo::nsm());
	model_template = pseudo::tnn(l_dims, neurons);
	models = new sync(model_template);
}

void att_lstm_ts::param_recur(function<void(Param&)> a_func) {
	model_template->param_recur(a_func);
}

model* att_lstm_ts::clone(function<Param(Param&)> a_func) {
	att_lstm_ts* result = new att_lstm_ts();
	result->units = units;
	result->htx = htx.clone();
	result->htx_grad = htx_grad.clone();
	result->model_template = model_template->clone(a_func);
	result->models = (sync*)models->clone(a_func);
	return result;
}

void att_lstm_ts::fwd() {
	models->fwd();
	y.clear();
	for (int i = 0; i < models->unrolled.size(); i++) {
		double att_factor = models->y[i][0].val();
		for (int j = 0; j < units; j++)
			y[j].val() += x[i][j].val() * att_factor;
	}
}

void att_lstm_ts::bwd() {
	for (int i = 0; i < models->unrolled.size(); i++) {
		double att_factor = models->y[i][0].val();
		double att_factor_grad = 0;
		for (int j = 0; j < units; j++)
			att_factor_grad += y_grad[j].val() * x[i][j].val();
		models->unrolled[i]->y_grad[0].val() = att_factor_grad;
	}
	models->bwd();

	htx_grad.clear();
	for (int i = 0; i < models->unrolled.size(); i++) {
		double att_factor = models->y[i][0].val();
		for (int j = 0; j < units; j++)
			x_grad[i][j].val() += y_grad[j].val() * att_factor;

		// LOOP THROUGH HIDDEN STATE INPUT GRADIENT
		for (int j = 0; j < units; j++)
			htx_grad[j].val() += models->unrolled[i]->x_grad[j];
	}
}

void att_lstm_ts::signal(const tensor& a_y_des) {
	y.sub_1d(a_y_des, y_grad);
}

void att_lstm_ts::model_recur(function<void(model*)> a_func) {
	model_template->model_recur(a_func);
}

void att_lstm_ts::compile() {
	models->compile();
	this->x = tensor::new_2d(models->prepared.size(), units);
	this->x_grad = tensor::new_2d(models->prepared.size(), units);
	this->y = tensor::new_1d(units);
	this->y_grad = tensor::new_1d(units);
	this->htx = tensor::new_1d(units);
	this->htx_grad = tensor::new_1d(units);
	for (int i = 0; i < models->prepared.size(); i++) {
		// cat ORDER: HT, **THEN** XT
		tensor htx_range = models->prepared[i]->x.range(0, units);
		tensor x_range = models->prepared[i]->x.range(units, units);
		tensor x_grad_range = models->prepared[i]->x_grad.range(units, units);
		htx.group_add_all_ranks(htx_range);
		x[i].group_join_all_ranks(x_range);
		x_grad[i].group_join_all_ranks(x_grad_range);
	}
}

void att_lstm_ts::prep(size_t a_n) {
	models->prep(a_n);
}

void att_lstm_ts::unroll(size_t a_n) {
	models->unroll(a_n);
}