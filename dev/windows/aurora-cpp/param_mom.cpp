#pragma once
#include "param.h"
#include "param_sgd.h"
#include "param_mom.h"

using aurora::params::param;
using aurora::params::param_sgd;
using aurora::params::param_mom;

param_mom::~param_mom() {

}

param_mom::param_mom() {

}

param_mom::param_mom(double a_state, double a_drop_chance, double a_learn_rate, double a_gradient, double a_momentum, double a_beta) : param_sgd(a_state, a_drop_chance, a_learn_rate, a_gradient) {
	this->momentum() = a_momentum;
	this->beta() = a_beta;
}

double& param_mom::momentum() {
	return momentum_ptr.val();
}

double& param_mom::beta() {
	return beta_ptr.val();
}

void param_mom::update() {
	if(!dropped) {
		momentum() = beta() * momentum() + (1 - beta()) * gradient();
		state() -= learn_rate() * momentum();
		gradient() = 0;
	}
	drop();
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