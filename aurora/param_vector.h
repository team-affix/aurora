#pragma once
#include "affix-base/pch.h"
#include "param.h"
#include "tensor.h"

namespace aurora {
	namespace params {
		class param_vector : public std::vector<Param> {
		protected:
			static std::uniform_real_distribution<double> s_urd;

		public:
			void pop(
				const aurora::maths::tensor& a_states
			);
			void randomize();
			void normalize();

		public:
			void update();

		public:
			operator aurora::maths::tensor() const;
			std::string to_string() const;

		};
	}
}
