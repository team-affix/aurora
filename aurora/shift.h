#pragma once
#include "affix-base/pch.h"
#include "model.h"

namespace aurora {
	namespace models {
		class shift : public model {
		public:
			size_t m_units = 0;

		public:
			std::vector<int> m_valid_shifts;

		public:
			aurora::maths::tensor m_amount;
			aurora::maths::tensor m_amount_grad;

		public:
			MODEL_FIELDS
			virtual ~shift();
			shift();
			shift(
				size_t a_units,
				std::vector<int> a_valid_shifts
			);

		protected:
			int positive_modulo(
				int i,
				int n
			);

		};
		typedef affix_base::data::ptr<shift> Shift;
	}
}
