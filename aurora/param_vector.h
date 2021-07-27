#pragma once
#include "pch.h"
#include "param.h"
#include "tensor.h"

using aurora::params::Param;
using aurora::maths::tensor;

namespace aurora {
	namespace params {
		class param_vector : public vector<Param> {
		public:
			void update();
			void pop(const tensor& a_states);

		public:
			operator tensor();

		};
	}
}

