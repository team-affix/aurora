#include "model.h"

using namespace aurora;
using namespace modeling;

model::model() {

}

model::model(vector<param>& pl) {

}

model::model(vector<param_sgd>& pl) {

}

model::model(vector<param_mom>& pl) {

}

void model::fwd() {
	y = x;
}

void model::bwd() {
	x_grad = y_grad;
}

void model::recur(function<void(model&)> func) {
	func(*this);
}

void model::prepend(model& _other) {
	x.link(_other.y);
	x_grad.link(_other.y_grad);
}

void model::append(model& _other) {
	y.link(_other.x);
	y_grad.link(_other.x_grad);
}