#pragma once
#include "affix-base/pch.h"
#include "model.h"

namespace aurora {
	namespace models {
		class interpolate : public model {
		public:
			size_t units = 0;

		public:
			double amount_compliment = 0;

		public:
			aurora::maths::tensor amount;
			aurora::maths::tensor amount_grad;

		public:
			MODEL_FIELDS
			virtual ~interpolate();
			interpolate();
			interpolate(size_t a_units);

		};
		typedef affix_base::data::ptr<interpolate> Interpolate;
	}
}
