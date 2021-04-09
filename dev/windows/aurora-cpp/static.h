#pragma once
#include <random>

using std::default_random_engine;
using std::uniform_real_distribution;

namespace aurora {
	class static_vals {
	public:
		static default_random_engine re;
		static uniform_real_distribution<double> urd;
		static double default_leaky_relu_m;
		static double default_drop_chance;

	};
}