#include "affix-base/pch.h"
#include "recurrent.h"

using aurora::models::recurrent;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::params::param_sgd;
using aurora::maths::tensor;
using std::vector;

recurrent::~recurrent() {

}

recurrent::recurrent() {

}

void recurrent::param_recur(function<void(Param&)> a_func) {

}

model* recurrent::clone(function<Param(Param&)> a_func) {
	recurrent* result = new recurrent();
	return result;
}

void recurrent::fwd() {

}

void recurrent::bwd() {

}

void recurrent::signal(const tensor& a_y_des) {
	
}

void recurrent::model_recur(function<void(model*)> a_func) {
	a_func(this);
}

void recurrent::compile() {

}

void recurrent::prep(size_t a_n) {

}

void recurrent::unroll(size_t a_n) {

}
