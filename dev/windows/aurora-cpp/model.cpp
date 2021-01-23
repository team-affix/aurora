#include "model.h"

using namespace aurora;
using namespace modeling;

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

void model::bwd() {

}

tensor& model::fwd(tensor a_x) {
	x.set(a_x);
	fwd();
	return y;
}

tensor& model::bwd(tensor a_y_grad) {
	y_grad.set(a_y_grad);
	bwd();
	return x_grad;
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

void model::compile() {
	y.group_add(x);
	y_grad.group_add(x_grad);
}