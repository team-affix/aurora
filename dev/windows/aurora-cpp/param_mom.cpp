#include "param_mom.h"

using aurora::optimization::param_mom;

param_mom::~param_mom() {

}

double& param_mom::momentum() {
	return momentum_ptr.val();
}

double& param_mom::beta() {
	return beta_ptr.val();
}

param* param_mom::clone() {
	param_mom* result = new param_mom();
	result->state() = state();
	result->learn_rate() = learn_rate();
	result->gradient() = gradient();
	result->momentum() = momentum();
	result->beta() = beta();
	return result;
}