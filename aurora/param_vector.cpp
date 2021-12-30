#include "affix-base/pch.h"
#include "param_vector.h"
#include "tensor.h"
#include "static_vals.h"

using aurora::params::param;
using aurora::params::param_vector;
using aurora::maths::tensor;

std::uniform_real_distribution<double> param_vector::s_urd(-1, 1);

void param_vector::pop(const tensor& a_states) {
	for (int i = 0; i < size(); i++)
		at(i)->state() = a_states[i].val();
}

void param_vector::randomize()
{
	pop(tensor::new_1d(size(), s_urd, static_vals::random_engine));
}

void param_vector::normalize()
{
	pop((operator tensor()).signed_norm_1d());
}

void param_vector::update() {
	for (int i = 0; i < size(); i++)
		at(i)->update();
}

param_vector::operator tensor() const
{
	tensor result = tensor::new_1d(size());
	for (int i = 0; i < size(); i++)
		result[i].val() = at(i)->state();
	return result;
}

std::string param_vector::to_string() const
{
	return (operator tensor()).to_string();
}
