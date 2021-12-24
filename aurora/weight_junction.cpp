#include "affix-base/pch.h"
#include "weight_junction.h"

using aurora::models::weight_junction;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::params::param_sgd;
using aurora::maths::tensor;
using std::vector;
using std::initializer_list;
using std::uniform_real_distribution;

weight_junction::~weight_junction() {

}

weight_junction::weight_junction() {

}

weight_junction::weight_junction(size_t a_a, size_t a_b) {
	this->a = a_a;
	this->b = a_b;
	for (int i = 0; i < a_a; i++)
		weight_sets.push_back(new weight_set(a_b));
}

void weight_junction::param_recur(function<void(Param&)> a_func) {
	for (int i = 0; i < weight_sets.size(); i++)
		weight_sets[i]->param_recur(a_func);
}

model* weight_junction::clone(function<Param(Param&)> a_func) {
	weight_junction* result = new weight_junction();
	result->a = a;
	result->b = b;
	for (int i = 0; i < weight_sets.size(); i++)
		result->weight_sets.push_back((weight_set*)weight_sets[i]->clone(a_func));
	return result;
}

void weight_junction::fwd() {
	y.clear();
	for (int i = 0; i < weight_sets.size(); i++) {
		weight_sets[i]->fwd();
		y.add_1d(weight_sets[i]->y, y);
	}
}

void weight_junction::bwd() {
	for (int i = 0; i < weight_sets.size(); i++)
		weight_sets[i]->bwd();
}

void weight_junction::signal(const tensor& a_y_des) {
	y.sub_1d(a_y_des, y_grad);
}

void weight_junction::model_recur(function<void(model*)> a_func) {
	a_func(this);
	for (int i = 0; i < weight_sets.size(); i++)
		weight_sets[i]->model_recur(a_func);
}

void weight_junction::compile() {
	x = tensor::new_1d(a);
	x_grad = tensor::new_1d(a);
	y = tensor::new_1d(b);
	y_grad = tensor::new_1d(b);
	for (int i = 0; i < weight_sets.size(); i++) {
		weight_sets[i]->compile();
		weight_sets[i]->x.group_add(x[i]);
		weight_sets[i]->x_grad.group_add(x_grad[i]);
		weight_sets[i]->y_grad.group_add(y_grad);
	}
}