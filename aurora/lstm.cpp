#include "pch.h"
#include "lstm.h"

using aurora::models::lstm;

lstm::~lstm() {

}

lstm::lstm() {

}

lstm::lstm(size_t a_units, size_t a_max_size, function<void(ptr<param>&)> a_init) {
	this->units = a_units;
	lstm_ts_template = new lstm_ts(units, a_init);
	prep(a_max_size);
}

lstm::lstm(size_t a_units, size_t a_max_size, ptr<lstm_ts> a_lstm_ts_template) {
	this->units = a_units;
	lstm_ts_template = a_lstm_ts_template;
	prep(a_max_size);
}

void lstm::pmt_wise(function<void(ptr<param>&)> a_func) {
	lstm_ts_template->pmt_wise(a_func);
}

model* lstm::clone() {
	return new lstm(units, prepared.size(), (lstm_ts*)lstm_ts_template->clone());
}

model* lstm::clone(function<void(ptr<param>&)> a_init) {
	return new lstm(units, prepared.size(), (lstm_ts*)lstm_ts_template->clone(a_init));
}

void lstm::fwd() {
	for (int i = 0; i < unrolled.size(); i++)
		unrolled[i]->fwd();
}

void lstm::bwd() {
	for (int i = unrolled.size() - 1; i >= 0; i--)
		unrolled[i]->bwd();
}

tensor& lstm::fwd(tensor& a_x) {
	x.pop(a_x);
	fwd();
	return y;
}

tensor& lstm::bwd(tensor& a_y_grad) {
	y_grad.pop(a_y_grad);
	bwd();
	return x_grad;
}

void lstm::signal(tensor& a_y_des) {
	for (int i = 0; i < unrolled.size(); i++)
		unrolled[i]->signal(a_y_des[i]);
}

void lstm::cycle(tensor& a_x, tensor& a_y_des) {
	x.pop(a_x);
	fwd();
	signal(a_y_des);
	bwd();
}

void lstm::recur(function<void(model*)> a_func) {
	a_func(this);
	lstm_ts_template->recur(a_func);
}

void lstm::compile() {
	this->x = tensor::new_2d(prepared.size(), units);
	this->y = tensor::new_2d(prepared.size(), units);
	this->x_grad = tensor::new_2d(prepared.size(), units);
	this->y_grad = tensor::new_2d(prepared.size(), units);
	this->ctx = tensor::new_1d(units);
	this->cty = tensor::new_1d(units);
	this->ctx_grad = tensor::new_1d(units);
	this->cty_grad = tensor::new_1d(units);
	this->htx = tensor::new_1d(units);
	this->hty = tensor::new_1d(units);
	this->htx_grad = tensor::new_1d(units);
	this->hty_grad = tensor::new_1d(units);
	tensor* ct = &ctx;
	tensor* ht = &htx;
	tensor* ct_grad = &ctx_grad;
	tensor* ht_grad = &htx_grad;
	for (int i = 0; i < prepared.size(); i++) {
		prepared[i]->compile();
		prepared[i]->x.group_add(x[i]);
		prepared[i]->y.group_add(y[i]);
		prepared[i]->x_grad.group_add(x_grad[i]);
		prepared[i]->y_grad.group_add(y_grad[i]);
		ct->group_add(prepared[i]->ctx);
		ht->group_add(prepared[i]->htx);
		ct_grad->group_add(prepared[i]->ctx_grad);
		ht_grad->group_add(prepared[i]->htx_grad);
		ct = &prepared[i]->cty;
		ht = &prepared[i]->hty;
		ct_grad = &prepared[i]->cty_grad;
		ht_grad = &prepared[i]->hty_grad;
	}
	ct->group_add(cty);
	ht->group_add(hty);
	ct_grad->group_add(cty_grad);
	ht_grad->group_add(hty_grad);
}

void lstm::prep(size_t a_n) {
	prepared.clear();
	prepared.resize(a_n);
	for (int i = 0; i < a_n; i++)
		prepared.at(i) = (lstm_ts*)lstm_ts_template->clone();
}

void lstm::unroll(size_t a_n) {
	unrolled.clear();
	unrolled.resize(a_n);
	for (int i = 0; i < a_n; i++)
		unrolled.at(i) = prepared.at(i);
}