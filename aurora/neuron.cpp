#include "affix-base/pch.h"
#include "neuron.h"
#include "bias.h"
#include "sigmoid.h"
#include "tanh.h"
#include "leaky_relu.h"
#include "leaky_rexu.h"

using namespace aurora;
using aurora::models::Sequential;
using aurora::models::sequential;
using aurora::models::bias;
using aurora::models::sigmoid;
using aurora::models::tanh;
using aurora::models::leaky_relu;
using aurora::models::leaky_rexu;

Sequential pseudo::nsm() {
	return new sequential{ new bias(), new sigmoid() };
}

Sequential pseudo::nth() {
	return new sequential{ new bias(), new aurora::models::tanh() };
}

Sequential pseudo::nth(double a_a, double a_b, double a_c) {
	return new sequential{ new bias(), new aurora::models::tanh(a_a, a_b, a_c) };
}

Sequential pseudo::nlr(double a_m) {
	return new sequential{ new bias(), new leaky_relu(a_m) };
}

Sequential pseudo::nlrexu(double a_k) {
	return new sequential{ new bias(), new leaky_rexu(a_k) };
}
