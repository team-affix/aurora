#include "pch.h"
#include "weight.h"

using aurora::models::weight;

weight::~weight() {
	
}

weight::weight() {

}

weight::weight(function<void(Param&)> a_func) {
	a_func(pmt);
}

void weight::param_recur(function<void(Param&)> a_func) {
	a_func(pmt);
}

model* weight::clone(function<Param(Param&)> a_func) {
	weight* result = new weight();
	result->pmt = a_func(pmt);
	return result;
}

void weight::fwd() {
	y.val() = x.val() * pmt->state();
}

void weight::bwd() {
	param_sgd* pmt_sgd = (param_sgd*)pmt.get();
	pmt_sgd->accum_grad(y_grad.val() * x.val());
	x_grad.val() = y_grad.val() * pmt->state();
}

void weight::signal(tensor& a_y_des) {
	y_grad.val() = y.val() - a_y_des.val();
}

void weight::model_recur(function<void(model*)> a_func) {
	a_func(this);
}

void weight::compile() {

}