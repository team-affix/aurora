#include "param.h"

using aurora::optimization::param;

double& param::state() {
	return state_ptr.val();
}

param::~param() {

}

double& param::learn_rate() {
	return learn_rate_ptr.val();
}

param* param::clone() {
	param* result = new param();
	result->state() = state();
	result->learn_rate() = learn_rate();
	return result;
}