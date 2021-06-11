#include "pch.h"
#include "model.h"

using namespace aurora;
using namespace models;

model::~model() {

}

model::model() {

}

model::model(function<void(ptr<param>&)> a_func) {
	
}

void model::fwd() {

}

void model::bwd() {

}

void model::signal(tensor& a_y_des) {
	
}

void model::recur(function<void(model*)> a_func) {
	a_func(this);
}

void model::pmt_wise(function<void(ptr<param>&)> a_func) {

}

model* model::clone() {
	return new model();
}

model* model::clone(function<void(ptr<param>&)> a_func) {
	return new model();
}

void model::compile() {
	y.group_add(x);
	y_grad.group_add(x_grad);
}