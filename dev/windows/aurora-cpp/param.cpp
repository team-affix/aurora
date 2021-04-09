#pragma once
#include "param.h"
#include "param_sgd.h"
#include "param_mom.h"
#include "static.h"

using aurora::params::param;
using aurora::params::param_sgd;
using aurora::params::param_mom;

uniform_real_distribution<double> param::drop_urd(0, 1);

param::~param() {

}

param::param() {

}

param::param(double a_state, double a_drop_chance) {
	this->state() = a_state;
	this->drop_chance() = a_drop_chance;
}

bool param::should_drop() {
	return drop_urd(static_vals::re) < drop_chance();
}

double& param::state() {
	return state_ptr.val();
}

double& param::dropped_state() {
	return dropped_state_ptr.val();
}

double& param::drop_chance() {
	return drop_chance_ptr.val();
}

void param::drop() {
	bool l_should_drop = should_drop();
	if (l_should_drop && !dropped) {
		dropped_state() = state();
		state() = 0;
		dropped = true;
	}
	else if (!l_should_drop && dropped) {
		state() = dropped_state();
		dropped_state() = 0;
		dropped = false;
	}
}

void param::update() {
	drop();
}

param* param::clone() {
	param* result = new param();
	result->state() = state();
	return result;
}