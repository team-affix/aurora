#include "pch.h"
#include "layer.h"

using aurora::models::layer;

layer::~layer() {

}

layer::layer() {

}

layer::layer(size_t a_a, Model a_model_template, function<void(ptr<param>&)> a_func) {
	for (size_t i = 0; i < a_a; i++)
		models.push_back(a_model_template->clone(a_func));
}

layer::layer(size_t a_a, Model a_model_template) {
	for (size_t i = 0; i < a_a; i++)
		models.push_back(a_model_template->clone());
}

layer::layer(initializer_list<Model> a_il) {
	std::copy(a_il.begin(), a_il.end(), back_inserter(models));
}

void layer::pmt_wise(function<void(ptr<param>&)> a_func) {
	for (int i = 0; i < models.size(); i++)
		models[i]->pmt_wise(a_func);
}

model* layer::clone() {
	layer* result = new layer();
	for (size_t i = 0; i < models.size(); i++)
		result->models.push_back(models[i]->clone());
	return result;
}

model* layer::clone(function<void(ptr<param>&)> a_func) {
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

void layer::recur(function<void(model*)> a_func) {
	a_func(this);
	for (size_t i = 0; i < models.size(); i++)
		models[i]->recur(a_func);
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