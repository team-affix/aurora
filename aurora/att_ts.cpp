#include "pch.h"
#include "att_ts.h"

using aurora::models::att_ts;

att_ts::~att_ts() {

}

att_ts::att_ts() {

}

att_ts::att_ts(size_t a_xt_units, size_t a_ht_units, vector<size_t> a_h_dims, function<void(ptr<param>&)> a_func) {
	this->xt_units = a_xt_units;
	this->ht_units = a_ht_units;
	vector<size_t> l_dims;
	l_dims.push_back(a_xt_units + a_ht_units);
	l_dims.insert(l_dims.end(), a_h_dims.begin(), a_h_dims.end());
	l_dims.push_back(1);
	// INITIALIZE NEURONS IN TNN
	vector<ptr<model>> neurons = vector<ptr<model>>(l_dims.size());
	for (int i = 0; i < l_dims.size() - 1; i++)
		neurons[i] = pseudo::nlr(0.3);
	neurons.push_back(pseudo::nsm());
	model_template = pseudo::tnn(l_dims, neurons, a_func);
	models = new sync(model_template);
}

void att_ts::pmt_wise(function<void(ptr<param>&)> a_func) {
	model_template->pmt_wise(a_func);
}

model* att_ts::clone() {
	att_ts* result = new att_ts();
	result->xt_units = xt_units;
	result->ht_units = ht_units;
	result->htx = htx.clone();
	result->htx_grad = htx_grad.clone();
	result->model_template = model_template->clone();
	result->models = (sync*)models->clone();
	return result;
}

model* att_ts::clone(function<void(ptr<param>&)> a_func) {
	att_ts* result = new att_ts();
	result->xt_units = xt_units;
	result->ht_units = ht_units;
	result->htx = htx.clone();
	result->htx_grad = htx_grad.clone();
	result->model_template = model_template->clone(a_func);
	result->models = (sync*)models->clone(a_func);
	return result;
}

void att_ts::fwd() {
	models->fwd();
	y.clear();
	for (int i = 0; i < models->unrolled.size(); i++) {
		double att_factor = models->y[i][0].val();
		for (int j = 0; j < xt_units; j++)
			y[j].val() += x[i][j].val() * att_factor;
	}
}

void att_ts::bwd() {
	for (int i = 0; i < models->unrolled.size(); i++) {
		double att_factor = models->y[i][0].val();
		double att_factor_grad = 0;
		for (int j = 0; j < xt_units; j++)
			att_factor_grad += y_grad[j].val() * x[i][j].val();
		models->unrolled[i]->y_grad[0].val() = att_factor_grad;
	}
	models->bwd();

	htx_grad.clear();
	for (int i = 0; i < models->unrolled.size(); i++) {
		double att_factor = models->y[i][0].val();
		for (int j = 0; j < xt_units; j++)
			x_grad[i][j].val() += y_grad[j].val() * att_factor;

		for (int j = 0; j < ht_units; j++)
			htx_grad[j].val() += models->unrolled[i]->x_grad[j];
	}
}

tensor& att_ts::fwd(tensor& a_x) {
	x.pop(a_x);
	fwd();
	return y;
}

tensor& att_ts::bwd(tensor& a_y_grad) {
	y_grad.pop(a_y_grad);
	bwd();
	return x_grad;
}

void att_ts::signal(tensor& a_y_des) {
	y.sub_1d(a_y_des, y_grad);
}

void att_ts::cycle(tensor& a_x, tensor& a_y_des) {
	x.pop(a_x);
	fwd();
	signal(a_y_des);
	bwd();
}

void att_ts::recur(function<void(model*)> a_func) {
	model_template->recur(a_func);
}

void att_ts::compile() {
	models->compile();
	this->x = tensor::new_2d(models->prepared.size(), xt_units);
	this->x_grad = tensor::new_2d(models->prepared.size(), xt_units);
	this->y = tensor::new_1d(xt_units);
	this->y_grad = tensor::new_1d(xt_units);
	this->htx = tensor::new_1d(ht_units);
	this->htx_grad = tensor::new_1d(ht_units);
	for (int i = 0; i < models->prepared.size(); i++) {
		// CONCAT ORDER: HT, **THEN** XT
		tensor htx_range = models->prepared[i]->x.range(0, ht_units);
		tensor x_range = models->prepared[i]->x.range(ht_units, xt_units);
		tensor x_grad_range = models->prepared[i]->x_grad.range(ht_units, xt_units);
		htx.group_add_all_ranks(htx_range);
		x[i].group_join_all_ranks(x_range);
		x_grad[i].group_join_all_ranks(x_grad_range);
	}
}

void att_ts::prep(size_t a_n) {
	models->prep(a_n);
}

void att_ts::unroll(size_t a_n) {
	models->unroll(a_n);
}