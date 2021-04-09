#include "static.h"

using aurora::static_vals;

default_random_engine static_vals::re;
uniform_real_distribution<double> static_vals::urd(-1, 1);
double static_vals::default_leaky_relu_m(0.3);
double static_vals::default_drop_chance(0.001);