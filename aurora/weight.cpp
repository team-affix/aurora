#include "pch.h"
#include "weight.h"

using aurora::models::weight;

weight::~weight() {
	
}

weight::weight() {

}

weight::weight(function<void(ptr<param>&)> a_init) {
	a_init(pmt);
}

void weight::pmt_wise(function<void(ptr<param>&)> a_func) {
	a_func(pmt);
}

model* weight::clone() {
	weight* result = new weight();
	result->pmt = pmt;
	return result;
}

model* weight::clone(function<void(ptr<param>&)> a_init) {
	weight* result = new weight();
	result->pmt = pmt->clone();
	a_init(result->pmt);
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

tensor& weight::fwd(tensor& a_x) {
	x.val() = a_x.val();
	fwd();
	return y;
}

tensor& weight::bwd(tensor& a_y_grad) {
	y_grad.val() = a_y_grad.val();
	bwd();
	return x_grad;
}

void weight::signal(tensor& a_y_des) {
	y_grad.val() = y.val() - a_y_des.val();
}

void weight::cycle(tensor& a_x, tensor& a_y_des) {
	x.val() = a_x.val();
	fwd();
	signal(a_y_des);
	bwd();
}

void weight::recur(function<void(model*)> a_func) {
	a_func(this);
}

void weight::compile() {

}