#include "pch.h"
#include "recurrent.h"

using aurora::models::recurrent;

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

void recurrent::signal(tensor& a_y_des) {
	
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
