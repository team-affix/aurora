#pragma once
#include "tensor.h"
#include "model.h"
#include "sequential.h"
#include "bias.h"
#include "weight.h"
#include "weight_set.h"
#include "weight_junction.h"
#include "layer.h"
#include <iostream>

using aurora::math::tensor;
using aurora::modeling::model;
using aurora::modeling::sequential;
using aurora::modeling::bias;
using aurora::modeling::weight;
using aurora::modeling::weight_set;
using aurora::modeling::weight_junction;
using aurora::modeling::layer;

int main() {

	vector<param_sgd*> pl = vector<param_sgd*>();

	layer l(10, new bias(), pl);
	l.compile();

	for (param_sgd* pmt : pl) {
		pmt->state() = 1;
	}

	l.x.set({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
	l.fwd();

	return 0;
}