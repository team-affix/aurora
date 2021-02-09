#include "bias.h"

using aurora::modeling::bias;

bias::~bias() {

}

bias::bias() {

}

bias::bias(function<void(ptr<param>&)> a_init) {
	a_init(pmt);
}

model* bias::clone() {
	bias* result = new bias();
	result->pmt = pmt;
	return result;
}

model* bias::clone(function<void(ptr<param>&)> a_init) {
	bias* result = new bias();
	result->pmt = pmt->clone();
	a_init(result->pmt);
	return result;
}

void bias::fwd() {
	y.val() = x.val() + pmt->state();
}

void bias::bwd() {
	param_sgd* pmt_sgd = (param_sgd*)pmt.get();
	pmt_sgd->accum_grad(y_grad.val());
}

tensor& bias::fwd(tensor a_x) {
	x.val() = a_x.val();
	fwd();
	return y;
}

tensor& bias::bwd(tensor a_y_grad) {
	y_grad.val() = a_y_grad.val();
	bwd();
	return x_grad;
}

void bias::signal(tensor a_y_des) {
	y_grad.val() = y.val() - a_y_des.val();
}

void bias::cycle(tensor a_x, tensor a_y_des) {
	x.val() = a_x.val();
	fwd();
	signal(a_y_des);
	bwd();
}

void bias::recur(function<void(model*)> a_func) {
	a_func(this);
}

void bias::compile() {
	x_grad.group_add(y_grad);
}