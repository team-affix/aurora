#include "pch.h"
#include "neuron.h"
#include "bias.h"
#include "sigmoid.h"
#include "tanh.h"
#include "leaky_relu.h"

using namespace aurora;
using aurora::models::bias;
using aurora::models::sigmoid;
using aurora::models::tanh;
using aurora::models::leaky_relu;

sequential* pseudo::nsm() {
	return new sequential{ new bias(), new sigmoid() };
}

sequential* pseudo::nth() {
	return new sequential{ new bias(), new aurora::models::tanh() };
}

sequential* pseudo::nlr(double a_m) {
	return new sequential{ new bias(), new leaky_relu(a_m) };
}