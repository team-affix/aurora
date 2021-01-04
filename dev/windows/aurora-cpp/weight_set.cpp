#include "weight_set.h"

using aurora::modeling::weight_set;

weight_set::~weight_set() {

}

weight_set::weight_set() {

}

weight_set::weight_set(size_t a, vector<param*>& pl) {
	weights.resize(a);
	for (size_t i = 0; i < a; i++)
		weights[i] = weight(pl);
}

weight_set::weight_set(size_t a, vector<param_sgd*>& pl) {
	weights.resize(a);
	for (size_t i = 0; i < a; i++)
		weights[i] = weight(pl);
}

weight_set::weight_set(size_t a, vector<param_mom*>& pl) {
	weights.resize(a);
	for (size_t i = 0; i < a; i++)
		weights[i] = weight(pl);
}

void weight_set::fwd() {
	for (size_t i = 0; i < weights.size(); i++)
		weights[i].fwd();
}

void weight_set::bwd() {
	x_grad.val() = 0;
	for (size_t i = 0; i < weights.size(); i++)
	{
		weights[i].bwd();
		x_grad.val() += weights[i].x_grad;
	}
}

void weight_set::recur(function<void(model&)> func) {
	func(*this);
	for (size_t i = 0; i < weights.size(); i++)
		weights[i].recur(func);
}

void weight_set::compile() {
	y = complex::new_1d(weights.size());
	for (size_t i = 0; i < weights.size(); i++) {
		weights[i].x.link(x);
		weights[i].y.link(y[i]);
	}
}