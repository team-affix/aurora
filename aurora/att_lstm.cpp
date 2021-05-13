#include "pch.h"
#include "att_lstm.h"

using aurora::models::att_lstm;

att_lstm::~att_lstm() {

}

att_lstm::att_lstm() {

}

att_lstm::att_lstm(size_t a_units, vector<size_t> a_h_dims, function<void(ptr<param>&)> a_func) {
	this->units = a_units;
	models = new sync(new att_lstm_ts(a_units, a_h_dims, a_func));
	internal_lstm = new lstm(a_units, a_func);
}

void att_lstm::pmt_wise(function<void(ptr<param>&)> a_func) {
	models->pmt_wise(a_func);
	internal_lstm->pmt_wise(a_func);
}

model* att_lstm::clone() {
	att_lstm* result = new att_lstm();
	result->units = units;
	result->models = (sync*)models->clone();
	result->internal_lstm = (lstm*)internal_lstm->clone();
	return result;
}

model* att_lstm::clone(function<void(ptr<param>&)> a_func) {
	att_lstm* result = new att_lstm();
	result->units = units;
	result->models = (sync*)models->clone(a_func);
	result->internal_lstm = (lstm*)internal_lstm->clone(a_func);
	return result;
}

void att_lstm::fwd() {
	for (int i = 0; i < models->unrolled.size(); i++) {
		models->unrolled[i]->fwd();
		internal_lstm->unrolled[i]->fwd();
	}
}

void att_lstm::bwd() {
	x_grad.clear();
	for (int i = models->unrolled.size() - 1; i >= 0; i--) {
		att_lstm_ts* ats = (att_lstm_ts*)models->unrolled[i].get();
		lstm_ts* lts = internal_lstm->unrolled[i].get();
		lts->bwd();
		ats->bwd();
		lts->htx_grad.add_1d(ats->htx_grad, lts->htx_grad);
		x_grad.add_2d(ats->x_grad, x_grad);
	}
}

tensor& att_lstm::fwd(tensor& a_x) {
	x.pop(a_x);
	fwd();
	return y;
}

tensor& att_lstm::bwd(tensor& a_y_grad) {
	y_grad.pop(a_y_grad);
	bwd();
	return x_grad;
}

void att_lstm::signal(tensor& a_y_des) {
	internal_lstm->signal(a_y_des);
}

void att_lstm::cycle(tensor& a_x, tensor& a_y_des) {
	x.pop(a_x);
	fwd();
	signal(a_y_des);
	bwd();
}

void att_lstm::recur(function<void(model*)> a_func) {
	models->recur(a_func);
	internal_lstm->recur(a_func);
}

void att_lstm::compile() {
	size_t l_a = models->prepared.size();
	size_t l_b = ((att_lstm_ts*)models->model_template.get())->models->prepared.size();
	this->x = tensor::new_2d(l_b, units);
	this->x_grad = tensor::new_2d(l_b, units);
	this->y = tensor::new_2d(l_a, units);
	this->y_grad = tensor::new_2d(l_a, units);
	models->compile();
	internal_lstm->compile();
	this->y.group_join(internal_lstm->y);
	this->y_grad.group_join(internal_lstm->y_grad);
	for (int i = 0; i < l_a; i++) {
		att_lstm_ts* ats = (att_lstm_ts*)models->prepared[i].get();
		lstm_ts* lts = internal_lstm->prepared[i].get();
		this->x.group_join(ats->x);
		ats->htx.group_join(lts->htx);
		lts->x.group_join(ats->y);
		lts->x_grad.group_join(ats->y_grad);
	}
}

void att_lstm::prep(size_t a_n, size_t b_n) {
	internal_lstm->prep(a_n);
	((att_lstm_ts*)models->model_template.get())->prep(b_n);
	models->prep(a_n);
}

void att_lstm::unroll(size_t a_n, size_t b_n) {
	internal_lstm->unroll(a_n);
	models->unroll(a_n);
	for (int i = 0; i < a_n; i++)
		((att_lstm_ts*)models->unrolled[i].get())->unroll(b_n);
}