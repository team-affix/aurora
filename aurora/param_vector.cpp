#include "affix-base/pch.h"
#include "param_vector.h"
#include "tensor.h"

using aurora::params::param;
using aurora::params::param_vector;
using aurora::maths::tensor;

void param_vector::update() {
	for (int i = 0; i < size(); i++)
		at(i)->update();
}

void param_vector::pop(const tensor& a_states) {
	for (int i = 0; i < size(); i++)
		at(i)->state() = a_states[i].val();
}

param_vector::operator tensor() {
	tensor result = tensor::new_1d(size());
	for (int i = 0; i < size(); i++)
		result[i].val() = at(i)->state();
	return result;
}
