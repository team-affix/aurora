#pragma once
#include "affix-base/pch.h"
#include "param.h"
#include "tensor.h"

namespace aurora {
	namespace params {
		class param_vector : public std::vector<Param> {
		public:
			void update();
			void pop(const aurora::maths::tensor& a_states);

		public:
			operator aurora::maths::tensor();

		};
	}
}
