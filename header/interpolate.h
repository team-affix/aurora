#pragma once
#include "pch.h"
#include "model.h"

using aurora::models::model;

namespace aurora {
	namespace models {
		class interpolate : public model {
		public:
			size_t units = 0;

		public:
			double amount_compliment = 0;

		public:
			tensor amount;
			tensor amount_grad;

		public:
			MODEL_FIELDS
			virtual ~interpolate();
			interpolate();
			interpolate(size_t a_units);

		};
	}
}
