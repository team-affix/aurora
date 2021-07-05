#pragma once
#include "pch.h"
#include "model.h"

using aurora::models::model;

namespace aurora {
	namespace models {
		class shift : public model {
		public:
			size_t units = 0;

		public:
			vector<int> valid_shifts;

		public:
			tensor amount;
			tensor amount_grad;

		public:
			MODEL_FIELDS
			virtual ~shift();
			shift();
			shift(size_t a_units, vector<int> a_valid_shifts);

		protected:
			int positive_modulo(int i, int n);

		};
		typedef ptr<shift> Shift;
	}
}