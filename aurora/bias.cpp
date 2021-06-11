#include "pch.h"
#include "bias.h"

using aurora::models::bias;

bias::~bias() {

}

bias::bias() {

}

bias::bias(function<void(ptr<param>&)> a_func) {
	a_func(pmt);
}

void bias::param_recur(function<void(ptr<param>&)> a_func) {
	a_func(pmt);
}

model* bias::clone() {
	bias* result = new bias();
	result->pmt = pmt;
	return result;
}

model* bias::clone(function<void(ptr<param>&)> a_func) {
	bias* result = new bias();
	result->pmt = pmt->clone();
	a_func(result->pmt);
	return result;
}

void bias::fwd() {
	y.val() = x.val() + pmt->state();
}

void bias::bwd() {
	param_sgd* pmt_sgd = (param_sgd*)pmt.get();
	pmt_sgd->accum_grad(y_grad.val());
}

void bias::signal(tensor& a_y_des) {
	y_grad.val() = y.val() - a_y_des.val();
}

void bias::model_recur(function<void(model*)> a_func) {
	a_func(this);
}

void bias::compile() {
	x_grad.group_add(y_grad);
}