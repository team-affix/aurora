#include "affix-base/pch.h"
#include "model.h"

using namespace aurora;
using namespace models;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::params::param_sgd;
using aurora::maths::tensor;
using std::vector;

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
	m_y.group_add(m_x);
	m_y_grad.group_add(m_x_grad);
}