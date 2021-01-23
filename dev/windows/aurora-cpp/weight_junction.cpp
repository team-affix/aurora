#include "weight_junction.h"

using aurora::modeling::weight_junction;

weight_junction::~weight_junction() {

}

weight_junction::weight_junction() {

}

weight_junction::weight_junction(size_t a_a, size_t a_b, vector<param*>& a_pl) {
	this->a = a_a;
	this->b = a_b;
	for (int i = 0; i < a_a; i++)
		weight_sets.push_back(new weight_set(a_b, a_pl));
}

weight_junction::weight_junction(size_t a_a, size_t a_b, vector<param_sgd*>& a_pl) {
	this->a = a_a;
	this->b = a_b;
	for (int i = 0; i < a_a; i++)
		weight_sets.push_back(new weight_set(a_b, a_pl));
}

weight_junction::weight_junction(size_t a_a, size_t a_b, vector<param_mom*>& a_pl) {
	this->a = a_a;
	this->b = a_b;
	for (int i = 0; i < a_a; i++)
		weight_sets.push_back(new weight_set(a_b, a_pl));
}

model* weight_junction::clone() {
	weight_junction* result = new weight_junction();
	result->a = a;
	result->b = b;
	for (int i = 0; i < weight_sets.size(); i++)
		result->weight_sets.push_back((weight_set*)weight_sets[i]->clone());
	return result;
}

model* weight_junction::clone(vector<param*>& a_pl) {
	weight_junction* result = new weight_junction();
	result->a = a;
	result->b = b;
	for (int i = 0; i < weight_sets.size(); i++)
		result->weight_sets.push_back((weight_set*)weight_sets[i]->clone(a_pl));
	return result;
}

model* weight_junction::clone(vector<param_sgd*>& a_pl) {
	weight_junction* result = new weight_junction();
	result->a = a;
	result->b = b;
	for (int i = 0; i < weight_sets.size(); i++)
		result->weight_sets.push_back((weight_set*)weight_sets[i]->clone(a_pl));
	return result;
}

model* weight_junction::clone(vector<param_mom*>& a_pl) {
	weight_junction* result = new weight_junction();
	result->a = a;
	result->b = b;
	for (int i = 0; i < weight_sets.size(); i++)
		result->weight_sets.push_back((weight_set*)weight_sets[i]->clone(a_pl));
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

tensor& weight_junction::fwd(tensor a_x) {
	x.pop(a_x);
	fwd();
	return y;
}

tensor& weight_junction::bwd(tensor a_y_grad) {
	y_grad.pop(a_y_grad);
	bwd();
	return x_grad;
}

void weight_junction::signal(tensor a_y_des) {
	y.sub_1d(a_y_des, y_grad);
}

void weight_junction::cycle(tensor a_x, tensor a_y_des) {
	x.pop(a_x);
	fwd();
	signal(a_y_des);
	bwd();
}

void weight_junction::recur(function<void(model*)> a_func) {
	a_func(this);
	for (int i = 0; i < weight_sets.size(); i++)
		weight_sets[i]->recur(a_func);
}

void weight_junction::compile() {
	x.resize(a);
	x_grad.resize(a);
	y.resize(b);
	y_grad.resize(b);
	for (int i = 0; i < weight_sets.size(); i++) {
		weight_sets[i]->compile();
		weight_sets[i]->x.group_add(x[i]);
		weight_sets[i]->x_grad.group_add(x_grad[i]);
		weight_sets[i]->y_grad.group_add(y_grad);
	}
}