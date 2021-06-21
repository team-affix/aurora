#include "pch.h"
#include "leaky_relu.h"

using aurora::models::leaky_relu;

leaky_relu::~leaky_relu() {

}

leaky_relu::leaky_relu(double a_m) {
	m.val() = a_m;
}

void leaky_relu::param_recur(function<void(Param&)> a_func) {

}

model* leaky_relu::clone(function<Param(Param&)> a_func) {
	return new leaky_relu(m.val());
}

void leaky_relu::fwd() {
	if (x.val() > 0)
		y.val() = x.val();
	else
		y.val() = m.val() * x.val();
}

void leaky_relu::bwd() {
	if (x.val() > 0)
		x_grad.val() = y_grad.val();
	else
		x_grad.val() = y_grad.val() * m.val();
}

void leaky_relu::signal(const tensor& a_y_des) {
	y_grad.val() = y.val() - a_y_des.val();
}

void leaky_relu::model_recur(function<void(model*)> a_func) {
	a_func(this);
}

void leaky_relu::compile() {

}