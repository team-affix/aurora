#pragma once
#include "pch.h"
#include "model.h"

using aurora::models::model;

namespace aurora {
	namespace models {
		class cos_sim : public model {
		public:
			size_t units = 0;

		public:
			double magnitude_0 = 0;
			double magnitude_1 = 0;
			double magnitude_product = 0;
			double dot_product = 0;

		public:
			MODEL_FIELDS
			virtual ~cos_sim();
			cos_sim();
			cos_sim(size_t a_units);

		};
		typedef ptr<cos_sim> Cos_sim;
	}
}
