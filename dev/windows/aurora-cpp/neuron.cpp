#include "neuron.h"
#include "bias.h"
#include "sigmoid.h"
#include "tanh.h"
#include "leaky_relu.h"

using namespace aurora;
using aurora::modeling::bias;
using aurora::modeling::sigmoid;
using aurora::modeling::tanh;
using aurora::modeling::leaky_relu;

sequential* pseudo::nsm() {
	return new sequential{ new bias(), new sigmoid() };
}

sequential* pseudo::nth() {
	return new sequential{ new bias(), new aurora::modeling::tanh() };
}

sequential* pseudo::nlr(double a_m) {
	return new sequential{ new bias(), new leaky_relu(a_m) };
}