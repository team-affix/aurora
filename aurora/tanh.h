#pragma once
#include "pch.h"
#include "model.h"

using aurora::models::model;

namespace aurora {
	namespace models {
		class tanh : public model {
		public:
			double a = 1;
			double b = 1;
			double c = 0;

		public:
			MODEL_FIELDS
			virtual ~tanh();
			tanh();
			tanh(double a_a, double a_b, double a_c);

		};
		typedef ptr<tanh> Tanh;
	}
}