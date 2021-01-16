#include "bias.h"

using aurora::modeling::bias;

bias::~bias() {

}

bias::bias() {

}

bias::bias(vector<param*>& a_pl) {
	pmt = new param();
	a_pl.push_back(pmt.get());
}

bias::bias(vector<param_sgd*>& a_pl) {
	pmt = new param_sgd();
	a_pl.push_back((param_sgd*)pmt.get());
}

bias::bias(vector<param_mom*>& a_pl) {
	pmt = new param_mom();
	a_pl.push_back((param_mom*)pmt.get());
}

model* bias::clone() {
	bias* result = new bias();
	result->pmt = pmt;
	return result;
}

model* bias::clone(vector<param*>& a_pl) {
	bias* result = new bias();
	result->pmt = pmt->clone();
	a_pl.push_back(result->pmt.get());
	return result;
}

model* bias::clone(vector<param_sgd*>& a_pl) {
	bias* result = new bias();
	result->pmt = pmt->clone();
	a_pl.push_back((param_sgd*)result->pmt.get());
	return result;
}

model* bias::clone(vector<param_mom*>& a_pl) {
	bias* result = new bias();
	result->pmt = pmt->clone();
	a_pl.push_back((param_mom*)result->pmt.get());
	return result;
}

void bias::fwd() {
	y.val() = x.val() + pmt->state();
}

void bias::bwd() {
	param_sgd* pmt_sgd = (param_sgd*)pmt.get();
	pmt_sgd->gradient() += y_grad.val();
}

tensor& bias::fwd(tensor a_x) {
	x.set(a_x);
	fwd();
	return y;
}

tensor& bias::bwd(tensor a_y_grad) {
	y_grad.set(a_y_grad);
	bwd();
	return x_grad;
}

void bias::signal(tensor a_y_des) {
	y_grad.val() = y.val() - a_y_des.val();
}

void bias::cycle(tensor a_x, tensor a_y_des) {
	x.set(a_x);
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