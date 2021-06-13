#include "pch.h"
#include "layer.h"

using aurora::models::layer;

layer::~layer() {

}

layer::layer() {

}

layer::layer(size_t a_num_models, Model a_model_template) {
	for (size_t i = 0; i < a_num_models; i++)
		models.push_back(a_model_template->clone());
}

layer::layer(vector<ptr<model>> a_models) {
	models = a_models;
}

void layer::param_recur(function<void(Param&)> a_func) {
	for (int i = 0; i < models.size(); i++)
		models[i]->param_recur(a_func);
}

model* layer::clone(function<Param(Param&)> a_func) {
	layer* result = new layer();
	for (size_t i = 0; i < models.size(); i++)
		result->models.push_back(models[i]->clone(a_func));
	return result;
}

void layer::fwd() {
	for (size_t i = 0; i < models.size(); i++)
		models[i]->fwd();
}

void layer::bwd() {
	for (size_t i = 0; i < models.size(); i++)
		models[i]->bwd();
}

void layer::signal(tensor& a_y_des) {
	for (size_t i = 0; i < models.size(); i++)
		models[i]->signal(a_y_des[i]);
}

void layer::model_recur(function<void(model*)> a_func) {
	a_func(this);
	for (size_t i = 0; i < models.size(); i++)
		models[i]->model_recur(a_func);
}

void layer::compile() {
	x.resize(models.size());
	y.resize(models.size());
	x_grad.resize(models.size());
	y_grad.resize(models.size());
	for (size_t i = 0; i < models.size(); i++) {
		models[i]->compile();
		models[i]->x.group_add(x[i]);
		models[i]->x_grad.group_add(x_grad[i]);
		models[i]->y.group_add(y[i]);
		models[i]->y_grad.group_add(y_grad[i]);
	}
}
