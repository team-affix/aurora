#include "weight.h"

using aurora::modeling::weight;

weight::~weight() {
	
}

weight::weight() {

}

weight::weight(vector<param*>& a_pl) {
	pmt = new param();
	a_pl.push_back(pmt.get());
}

weight::weight(vector<param_sgd*>& a_pl) {
	pmt = new param_sgd();
	a_pl.push_back((param_sgd*)pmt.get());
}

weight::weight(vector<param_mom*>& a_pl) {
	pmt = new param_mom();
	a_pl.push_back((param_mom*)pmt.get());
}

model* weight::clone() {
	weight* result = new weight();
	result->pmt = pmt;
	return result;
}

model* weight::clone(vector<param*>& a_pl) {
	weight* result = new weight();
	result->pmt = pmt->to_param();
	a_pl.push_back(result->pmt.get());
	return result;
}

model* weight::clone(vector<param_sgd*>& a_pl) {
	weight* result = new weight();
	result->pmt = pmt->to_param_sgd();
	a_pl.push_back((param_sgd*)result->pmt.get());
	return result;
}

model* weight::clone(vector<param_mom*>& a_pl) {
	weight* result = new weight();
	result->pmt = pmt->to_param_mom();
	a_pl.push_back((param_mom*)result->pmt.get());
	return result;
}

void weight::fwd() {
	y.val() = x.val() * pmt->state();
}

void weight::bwd() {
	param_sgd* pmt_sgd = (param_sgd*)pmt.get();
	pmt_sgd->gradient() += y_grad.val() * x.val();
	x_grad.val() = y_grad.val() * pmt->state();
}

tensor& weight::fwd(tensor a_x) {
	x.val() = a_x.val();
	fwd();
	return y;
}

tensor& weight::bwd(tensor a_y_grad) {
	y_grad.val() = a_y_grad.val();
	bwd();
	return x_grad;
}

void weight::signal(tensor a_y_des) {
	y_grad.val() = y.val() - a_y_des.val();
}

void weight::cycle(tensor a_x, tensor a_y_des) {
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