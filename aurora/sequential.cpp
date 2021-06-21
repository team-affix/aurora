#include "pch.h"
#include "sequential.h"

using aurora::models::sequential;

sequential::~sequential() {

}

sequential::sequential() {

}

sequential::sequential(initializer_list<Model> a_models) {
	for (initializer_list<Model>::iterator i = a_models.begin(); i != a_models.end(); i++)
		models.push_back(*i);
}

sequential::sequential(vector<Model> a_models) {
	models = a_models;
}

void sequential::param_recur(function<void(Param&)> a_func) {
	for (int i = 0; i < models.size(); i++)
		models[i]->param_recur(a_func);
}

model* sequential::clone(function<Param(Param&)> a_func) {
	sequential* result = new sequential();
	result->models.resize(models.size());
	for (int i = 0; i < models.size(); i++)
		result->models[i] = models[i]->clone(a_func);
	return result;
}

void sequential::fwd() {
	for (int i = 0; i < models.size(); i++)
		models[i]->fwd();
}

void sequential::bwd() {
	for (int i = models.size() - 1; i >= 0; i--) {
		models[i]->bwd();
	}
}

void sequential::signal(const tensor& a_y_des) {
	models.back()->signal(a_y_des);
}

void sequential::model_recur(function<void(model*)> a_func) {
	a_func(this);
	for (int i = 0; i < models.size(); i++)
		models[i]->model_recur(a_func);
}

void sequential::compile() {
	tensor* state = &x;
	tensor* gradient = &x_grad;
	for (int i = 0; i < models.size(); i++) {
		models[i]->compile();
		models[i]->x.group_add_all_ranks(*state);
		models[i]->x_grad.group_add_all_ranks(*gradient);
		state = &models[i]->y;
		gradient = &models[i]->y_grad;
	}
	state->group_add_all_ranks(y);
	gradient->group_add_all_ranks(y_grad);
}