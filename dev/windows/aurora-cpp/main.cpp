#pragma once
#include "complex.h"
#include "bias.h"
#include "weight_set.h"
#include <iostream>

using aurora::math::complex;
using aurora::modeling::bias;
using aurora::modeling::weight_set;


int main() {

	vector<param*> pl = vector<param*>();

	weight_set w(3, pl);

	return 0;
}