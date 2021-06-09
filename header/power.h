#pragma once
#include "pch.h"
#include "model.h"

using aurora::models::model;

namespace aurora {
	namespace models {
		class power : public model {
		public:
			size_t units = 0;

		public:
			tensor amount;
			tensor amount_grad;

		public:
			MODEL_FIELDS
			virtual ~power();
			power();
			power(size_t a_units);

		};
	}
}