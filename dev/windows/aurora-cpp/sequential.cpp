#include "sequential.h"

using aurora::modeling::sequential;

sequential::~sequential() {

}

sequential::sequential() {

}

sequential::sequential(initializer_list<ptr<model>> a_il) {
	models.resize(a_il.size());
	for (int i = 0; i < a_il.size(); i++) {
		const ptr<model>* elem = a_il.begin() + i;
		models[i] = *elem;
	}
}

model* sequential::clone() {
	sequential* result = new sequential();
	result->models.resize(models.size());
	for (int i = 0; i < models.size(); i++)
		result->models[i] = models[i]->clone();
	return result;
}

model* sequential::clone(vector<param*>& a_pl) {
	sequential* result = new sequential();
	result->models.resize(models.size());
	for (int i = 0; i < models.size(); i++)
		result->models[i] = models[i]->clone(a_pl);
	return result;
}

model* sequential::clone(vector<param_sgd*>& a_pl) {
	sequential* result = new sequential();
	result->models.resize(models.size());
	for (int i = 0; i < models.size(); i++)
		result->models[i] = models[i]->clone(a_pl);
	return result;
}

model* sequential::clone(vector<param_mom*>& a_pl) {
	sequential* result = new sequential();
	result->models.resize(models.size());
	for (int i = 0; i < models.size(); i++)
		result->models[i] = models[i]->clone(a_pl);
	return result;
}

void sequential::fwd() {
	for (int i = 0; i < models.size(); i++)
		models[i]->fwd();
}

tensor& sequential::fwd(tensor a_x) {
	x() = a_x;
	fwd();
	return y();
}

void sequential::bwd() {
	for (int i = models.size() - 1; i >= 0; i--) {
		models[i]->bwd();
	}
}

tensor& sequential::bwd(tensor a_y_grad) {
	y_grad() = a_y_grad;
	bwd();
	return x_grad();
}

void sequential::signal(tensor a_y_des) {
	models.back()->signal(a_y_des);
}

void sequential::cycle(tensor a_x, tensor a_y_des) {
	x() = a_x;
	fwd();
	signal(a_y_des);
	bwd();
}

void sequential::recur(function<void(model*)> a_func) {
	a_func(this);
	for (int i = 0; i < models.size(); i++)
		models[i]->recur(a_func);
}

void sequential::compile() {
	ptr<tensor> l_x_ptr = m_x_ptr;
	ptr<tensor> l_x_grad_ptr = m_x_grad_ptr;
	for (int i = 0; i < models.size(); i++) {
		models[i]->compile();
		models[i]->m_x_ptr.link(l_x_ptr);
		models[i]->m_x_grad_ptr.link(l_x_grad_ptr);
		l_x_ptr = models[i]->m_y_ptr;
		l_x_grad_ptr = models[i]->m_y_grad_ptr;
	}
	m_y_ptr.link(l_x_ptr);
	m_y_grad_ptr.link(l_x_grad_ptr);
}