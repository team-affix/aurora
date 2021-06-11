#include "pch.h"
#include "leaky_rexu.h"

using aurora::models::leaky_rexu;

leaky_rexu::~leaky_rexu() {

}

leaky_rexu::leaky_rexu() {

}

leaky_rexu::leaky_rexu(double a_k) {
	k = a_k;
}

void leaky_rexu::param_recur(function<void(Param&)> a_func) {

}

model* leaky_rexu::clone(function<Param(Param&)> a_func) {
	leaky_rexu* result = new leaky_rexu();
	result->k = k;
	return result;
}

void leaky_rexu::fwd() {
	if (x < 0)
		y.val() = exp(x) + k_minus_one;
	else
		y.val() = x + k;
}

void leaky_rexu::bwd() {
	if (x < 0)
		x_grad.val() = y_grad * (y - k_minus_one);
	else
		x_grad.val() = y_grad;
}

void leaky_rexu::signal(tensor& a_y_des) {
	y_grad.val() = y - a_y_des;
}

void leaky_rexu::model_recur(function<void(model*)> a_func) {
	a_func(this);
}

void leaky_rexu::compile() {
	k_minus_one = k - 1.0;
}