#pragma once
#include "affix-base/pch.h"
#include "model.h"

namespace aurora {
	namespace models {
		class shift : public model {
		public:
			size_t units = 0;

		public:
			std::vector<int> valid_shifts;

		public:
			aurora::maths::tensor amount;
			aurora::maths::tensor amount_grad;

		public:
			MODEL_FIELDS
			virtual ~shift();
			shift();
			shift(size_t a_units, std::vector<int> a_valid_shifts);

		protected:
			int positive_modulo(int i, int n);

		};
		typedef affix_base::data::ptr<shift> Shift;
	}
}
