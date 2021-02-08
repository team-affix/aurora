#include "leaky_relu.h"

using aurora::modeling::leaky_relu;

leaky_relu::~leaky_relu() {

}

leaky_relu::leaky_relu(double a_m) {
	m.val() = a_m;
}

model* leaky_relu::clone() {
	return new leaky_relu(m.val());
}

model* leaky_relu::clone(function<void(ptr<param>&)> a_init) {
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

tensor& leaky_relu::fwd(tensor a_x) {
	x.val() = a_x.val();
	fwd();
	return y;
}

tensor& leaky_relu::bwd(tensor a_y_grad) {
	y_grad.val() = a_y_grad.val();
	bwd();
	return x_grad;
}

void leaky_relu::signal(tensor a_y_des) {
	y_grad.val() = y.val() - a_y_des.val();
}

void leaky_relu::cycle(tensor a_x, tensor a_y_des) {
	x.val() = a_x.val();
	fwd();
	signal(a_y_des);
	bwd();
}

void leaky_relu::recur(function<void(model*)> a_func) {
	a_func(this);
}

void leaky_relu::compile() {

}