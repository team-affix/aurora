#include "affix-base/pch.h"
#include "sync.h"

using aurora::models::sync;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::params::param_sgd;
using aurora::maths::tensor;
using std::vector;
using std::initializer_list;
using std::uniform_real_distribution;

sync::~sync() {

}

sync::sync(Model a_model_template) {
	this->model_template = a_model_template;
}

void sync::param_recur(function<void(Param&)> a_func) {
	model_template->param_recur(a_func);
}

model* sync::clone(function<Param(Param&)> a_func) {
	sync* result = new sync(model_template->clone(a_func));
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

void sync::signal(const tensor& a_y_des) {
	for (int i = 0; i < unrolled.size(); i++)
		unrolled[i]->signal(a_y_des[i]);
}

void sync::model_recur(function<void(model*)> a_func) {
	a_func(this);
	model_template->model_recur(a_func);
}

void sync::compile() {
	x.resize(prepared.size());
	y.resize(prepared.size());
	x_grad.resize(prepared.size());
	y_grad.resize(prepared.size());
	for (int i = 0; i < prepared.size(); i++) {
		prepared[i]->compile();
		x[i].group_join_all_ranks(prepared[i]->x);
		y[i].group_join_all_ranks(prepared[i]->y);
		x_grad[i].group_join_all_ranks(prepared[i]->x_grad);
		y_grad[i].group_join_all_ranks(prepared[i]->y_grad);
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
