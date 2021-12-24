#include "affix-base/pch.h"
#include "weight_set.h"

using aurora::models::weight_set;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::params::param_sgd;
using aurora::maths::tensor;
using std::vector;
using std::initializer_list;
using std::uniform_real_distribution;

weight_set::~weight_set() {

}

weight_set::weight_set() {

}

weight_set::weight_set(size_t a_a) {
	this->a = a_a;
	for (int i = 0; i < a_a; i++)
		weights.push_back(new weight());
}

void weight_set::param_recur(function<void(Param&)> a_func) {
	for (int i = 0; i < weights.size(); i++)
		weights[i]->param_recur(a_func);
}

model* weight_set::clone(function<Param(Param&)> a_func) {
	weight_set* result = new weight_set();
	result->a = a;
	for (int i = 0; i < a; i++)
		result->weights.push_back((weight*)weights[i]->clone(a_func));
	return result;
}

void weight_set::fwd() {
	for (int i = 0; i < weights.size(); i++)
		weights[i]->fwd();
}

void weight_set::bwd() {
	x_grad.val() = 0;
	for (int i = 0; i < weights.size(); i++) {
		weights[i]->bwd();
		x_grad.val() += weights[i]->x_grad.val();
	}
}

void weight_set::signal(const tensor& a_y_des) {
	y.sub_1d(a_y_des, y_grad);
}

void weight_set::model_recur(function<void(model*)> a_func) {
	a_func(this);
	for (int i = 0; i < weights.size(); i++)
		weights[i]->model_recur(a_func);
}

void weight_set::compile() {
	y.resize(a);
	y_grad.resize(a);
	for (int i = 0; i < a; i++) {
		weights[i]->compile();
		weights[i]->x.group_add(x);
		weights[i]->y.group_add(y[i]);
		weights[i]->y_grad.group_add(y_grad[i]);
	}
}