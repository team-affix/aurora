#include "param_sgd.h"

using aurora::optimization::param_sgd;

double& param_sgd::gradient() {
	return gradient_ptr.val();
}

param_sgd::~param_sgd() {

}

param* param_sgd::clone() {
	param_sgd* result = new param_sgd();
	result->state() = state();
	result->learn_rate() = learn_rate();
	result->gradient() = gradient();
	return result;
}