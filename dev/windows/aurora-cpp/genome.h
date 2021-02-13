#pragma once
#include "tensor.h"

using aurora::math::tensor;

namespace aurora {
	namespace optimization {
		class genome : public tensor {
		public:
			function<genome(genome&)> mutate;
			function<double(genome&)> get_cost;

		public:
			genome(tensor a_tens, function<genome(genome&)> a_mutate, function<double(genome&)> a_get_cost);

		};
	}
}