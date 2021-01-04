#include "weight.h"

using aurora::modeling::weight;

weight::~weight() {

}

weight::weight() {

}

weight::weight(vector<param*>& _pl) {
	pmt = new param();
	_pl.push_back(pmt);
}

weight::weight(vector<param_sgd*>& _pl) {
	pmt = new param_sgd();
	_pl.push_back((param_sgd*)pmt);
}

weight::weight(vector<param_mom*>& _pl) {
	pmt = new param_mom();
	_pl.push_back((param_mom*)pmt);
}

void weight::fwd() {
	y = x * pmt->state;
}

void weight::bwd() {
	param_sgd* pmt_sgd = (param_sgd*)pmt;
	x_grad.val() = y_grad * pmt_sgd->state;
	pmt_sgd->gradient += y_grad * x;
}

void weight::recur(function<void(model&)> func) {
	func(*this);
}

void weight::compile() {

}