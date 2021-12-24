#include "affix-base/pch.h"
#include "stacked_recurrent.h"

using aurora::models::stacked_recurrent;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::params::param_sgd;
using aurora::maths::tensor;
using std::vector;
using std::initializer_list;
using std::uniform_real_distribution;

stacked_recurrent::~stacked_recurrent() {

}

stacked_recurrent::stacked_recurrent() {

}

stacked_recurrent::stacked_recurrent(vector<Recurrent> a_models) {
	models = a_models;
}

stacked_recurrent::stacked_recurrent(size_t a_height, Recurrent a_model_template) {
	for (int i = 0; i < a_height; i++)
		models.push_back((recurrent*)a_model_template->clone());
}

void stacked_recurrent::param_recur(function<void(Param&)> a_func) {
	for (int i = 0; i < models.size(); i++)
		models[i]->param_recur(a_func);
}

model* stacked_recurrent::clone(function<Param(Param&)> a_func) {
	stacked_recurrent* result = new stacked_recurrent();
	for (int i = 0; i < models.size(); i++)
		result->models.push_back((recurrent*)models[i]->clone(a_func));
	result->prep(prepared_size);
	result->unroll(unrolled_size);
	return result;
}

void stacked_recurrent::fwd() {
	for (int i = 0; i < models.size(); i++)
		models[i]->fwd();
}

void stacked_recurrent::bwd() {
	for (int i = models.size() - 1; i >= 0; i--)
		models[i]->bwd();
}

void stacked_recurrent::signal(const tensor& a_y_des) {
	models.back()->signal(a_y_des);
}

void stacked_recurrent::model_recur(function<void(model*)> a_func) {
	for (int i = 0; i < models.size(); i++)
		models[i]->model_recur(a_func);
}

void stacked_recurrent::compile() {
	x = tensor::new_1d(prepared_size);
	x_grad = tensor::new_1d(prepared_size);
	y = tensor::new_1d(prepared_size);
	y_grad = tensor::new_1d(prepared_size);

	tensor* l_x = &x;
	tensor* l_x_grad = &x_grad;

	for (int i = 0; i < models.size(); i++) {
		models[i]->compile();
		l_x->group_join(models[i]->x);
		l_x_grad->group_join(models[i]->x_grad);
		l_x = &models[i]->y;
		l_x_grad = &models[i]->y_grad;
	}

	y.group_join(*l_x);
	y_grad.group_join(*l_x_grad);
}

void stacked_recurrent::prep(size_t a_n) {
	prepared_size = a_n;
	for (int i = 0; i < models.size(); i++)
		models[i]->prep(a_n);
}

void stacked_recurrent::unroll(size_t a_n) {
	unrolled_size = a_n;
	for (int i = 0; i < models.size(); i++)
		models[i]->unroll(a_n);
}