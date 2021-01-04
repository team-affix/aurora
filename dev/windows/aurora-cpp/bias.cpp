#include "bias.h"

using aurora::modeling::bias;

bias::~bias() {
	delete pmt;
}

bias::bias() {

}

bias::bias(vector<param*>& _pl) {
	pmt = new param();
	_pl.push_back(pmt);
}

bias::bias(vector<param_sgd*>& _pl) {
	pmt = new param_sgd();
	_pl.push_back((param_sgd*)pmt);
}

bias::bias(vector<param_mom*>& _pl) {
	pmt = new param_mom();
	_pl.push_back((param_mom*)pmt);
}

void bias::fwd() {
	y.val() = x.val() + pmt->state;
}

void bias::bwd() {
	param_sgd* pmt_sgd = (param_sgd*)pmt;
	x_grad.val() = y_grad.val();
	pmt_sgd->gradient += y_grad;
}

void bias::recur(function<void(model&)> func) {
	func(*this);
}

void bias::compile() {

}