#include "model.h"

using namespace aurora;
using namespace modeling;

tensor& model::x() {
	return x_ptr.val();
}

tensor& model::y() {
	return y_ptr.val();
}

tensor& model::x_grad() {
	return x_grad_ptr.val();
}

tensor& model::y_grad() {
	return y_grad_ptr.val();
}

model::~model() {

}

model::model() {

}

model::model(vector<param*>& pl) {

}

model::model(vector<param_sgd*>& pl) {

}

model::model(vector<param_mom*>& pl) {

}

void model::fwd() {

}

void model::bwd() {

}

void model::recur(function<void(model*)> _func) {
	_func(this);
}

model* model::clone() {
	return new model();
}

model* model::clone(vector<param*>& _pl) {
	return new model();
}

model* model::clone(vector<param_sgd*>& _pl) {
	return new model();
}

model* model::clone(vector<param_mom*>& _pl) {
	return new model();
}

void model::prepend(model& _other) {
	x_ptr.link(_other.y_ptr);
	x_grad_ptr.link(_other.y_grad_ptr);
	x = x_ptr.get();
	x_grad = x_grad_ptr.get();
}

void model::append(model& _other) {
	y_ptr.link(_other.x_ptr);
	y_grad_ptr.link(_other.x_grad_ptr);
	y = y_ptr.get();
	y_grad = y_grad_ptr.get();
}

void model::compile() {
	y_ptr.link(x_ptr);
	x_grad_ptr.link(y_grad_ptr);
	x = x_ptr.get();
	x_grad = x_grad_ptr.get();
	y = y_ptr.get();
	y_grad = y_grad_ptr.get();
}