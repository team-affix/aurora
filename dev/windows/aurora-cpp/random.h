#pragma once
#include <random>
#include <math.h>

namespace aurora {
	namespace math {
		class random {
		public:
			std::default_random_engine dre;

		public:
			random(unsigned int a_init);

		public:
			double next_double(double a_min, double a_max);

		};
	}
}