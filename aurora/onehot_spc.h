#pragma once
#include "affix-base/pch.h"
#include "model.h"
#include "normalize.h"

namespace aurora {
	namespace models {
		class onehot_spc : public model {
		protected:
			static std::uniform_real_distribution<double> s_urd;

		public:
			size_t m_units = 0;

		public:
			MODEL_FIELDS
			virtual ~onehot_spc();
			onehot_spc();
			onehot_spc(
				size_t a_units
			);

		protected:
			int collapse(
				const aurora::maths::tensor& a_probability_tensor
			);

		};
	}
}
