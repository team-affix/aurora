#pragma once
#include "affix-base/pch.h"
#include "model.h"
#include "normalize.h"

namespace aurora {
	namespace models {
		class spc : public model {
		protected:
			static std::uniform_real_distribution<double> m_urd;

		public:
			size_t units = 0;

		public:
			MODEL_FIELDS
			virtual ~spc();
			spc();
			spc(size_t a_units);

		protected:
			int collapse(const aurora::maths::tensor& a_probability_tensor);

		};
	}
}
