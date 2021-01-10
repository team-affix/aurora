#include "model.h"

using namespace aurora;
using namespace modeling;

tensor& model::x() {
	return m_x_ptr.val();
}

tensor& model::y() {
	return m_y_ptr.val();
}

tensor& model::x_grad() {
	return m_x_grad_ptr.val();
}

tensor& model::y_grad() {
	return m_y_grad_ptr.val();
}

model::~model() {

}

model::model() {

}

model::model(vector<param*>& a_pl) {

}

model::model(vector<param_sgd*>& a_pl) {

}

model::model(vector<param_mom*>& a_pl) {

}

void model::fwd() {

}

tensor& model::fwd(tensor a_x) {
	x() = a_x;
	fwd();
	return y();
}

void model::bwd() {

}

tensor& model::bwd(tensor a_y_grad) {
	y_grad() = a_y_grad;
	bwd();
	return x_grad();
}


void model::signal(tensor a_y_des) {
	
}

void model::cycle(tensor a_x, tensor a_y_des) {

}

void model::recur(function<void(model*)> a_func) {
	a_func(this);
}

model* model::clone() {
	return new model();
}

model* model::clone(vector<param*>& a_pl) {
	return new model();
}

model* model::clone(vector<param_sgd*>& a_pl) {
	return new model();
}

model* model::clone(vector<param_mom*>& a_pl) {
	return new model();
}

void model::append(model* a_other) {
	a_other->m_x_ptr.link(m_y_ptr);
	a_other->m_x_grad_ptr.link(m_y_grad_ptr);
}

void model::prepend(model* a_other) {
	a_other->m_y_ptr.link(m_x_ptr);
	a_other->m_y_grad_ptr.link(m_x_grad_ptr);
}

void model::compile() {
	m_y_ptr.link(m_x_ptr);
	m_y_grad_ptr.link(m_x_grad_ptr);
}