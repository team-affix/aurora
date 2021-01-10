#pragma once
#include "tensor.h"
#include "model.h"
#include "sequential.h"
#include <iostream>

using aurora::math::tensor;
using aurora::modeling::model;
using aurora::modeling::sequential;

int main() {

	model* m1 = new model();
	model* m2 = new model();

	sequential s = { m1, m2 };

	s.compile();

	tensor y = s.fwd({ 1, 2, 3 });
	
	return 0;
}