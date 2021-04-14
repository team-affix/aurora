#include "sync.h"

using aurora::models::sync;

sync::~sync() {

}

sync::sync(ptr<model> a_model_template) {
	this->model_template = a_model_template;
}

sync::sync(size_t a_max_size, ptr<model> a_model_template) {
	this->model_template = a_model_template;
	prep(a_max_size);
}

void sync::pmt_wise(function<void(ptr<param>&)> a_func) {
	model_template->pmt_wise(a_func);
}

model* sync::clone() {
	sync* result = new sync(prepared.size(), model_template->clone());
	result->prep(prepared.size());
	result->unroll(unrolled.size());
	return result;
}

model* sync::clone(function<void(ptr<param>&)> a_init) {
	sync* result = new sync(prepared.size(), model_template->clone(a_init));
	result->prep(prepared.size());
	result->unroll(unrolled.size());
	return result;
}

void sync::fwd() {
	for (int i = 0; i < unrolled.size(); i++)
		unrolled[i]->fwd();
}

void sync::bwd() {
	for (int i = 0; i < unrolled.size(); i++)
		unrolled[i]->bwd();
}

tensor& sync::fwd(tensor& a_x) {
	x.pop(a_x);
	fwd();
	return y;
}

tensor& sync::bwd(tensor& a_y_grad) {
	y_grad.pop(a_y_grad);
	bwd();
	return x_grad;
}

void sync::signal(tensor& a_y_des) {
	for (int i = 0; i < unrolled.size(); i++)
		unrolled[i]->signal(a_y_des[i]);
}

void sync::cycle(tensor& a_x, tensor& a_y_des) {
	x.pop(a_x);
	fwd();
	signal(a_y_des);
	bwd();
}

void sync::recur(function<void(model*)> a_func) {
	a_func(this);
	model_template->recur(a_func);
}

void sync::compile() {
	x.resize(prepared.size());
	y.resize(prepared.size());
	x_grad.resize(prepared.size());
	y_grad.resize(prepared.size());
	for (int i = 0; i < prepared.size(); i++) {
		prepared[i]->compile();
		x[i].group_join(prepared[i]->x);
		y[i].group_join(prepared[i]->y);
		x_grad[i].group_join(prepared[i]->x_grad);
		y_grad[i].group_join(prepared[i]->y_grad);
	}
}

void sync::prep(size_t a_n) {
	prepared.clear();
	prepared.resize(a_n);
	for (int i = 0; i < a_n; i++)
		prepared.at(i) = model_template->clone();
}

void sync::unroll(size_t a_n) {
	unrolled.clear();
	unrolled.resize(a_n);
	for (int i = 0; i < a_n; i++)
		unrolled.at(i) = prepared.at(i);
}
