#pragma once
#include "tensor.h"
#include "model.h"
#include <iostream>

using aurora::math::tensor;
using aurora::modeling::model;

int main() {

	model m1, m2 = model();
	m1.append(m2);

	m1.compile();
	m2.compile();



	return 0;
}