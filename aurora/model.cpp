#include "pch.h"
#include "model.h"

using namespace aurora;
using namespace models;

model::~model() {

}

model::model() {

}

void model::fwd() {

}

void model::bwd() {

}

void model::signal(const tensor& a_y_des) {
	
}

void model::model_recur(function<void(model*)> a_func) {
	a_func(this);
}

void model::param_recur(function<void(Param&)> a_func) {

}

model* model::clone(function<Param(Param&)> a_func) {
	return new model();
}

void model::compile() {
	y.group_add(x);
	y_grad.group_add(x_grad);
}