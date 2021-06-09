#pragma once
#include "pch.h"
#include "model.h"

using aurora::models::model;

namespace aurora {
	namespace models {
		class leaky_rexu : public model {
		public:
			double k = 0;
			double k_minus_one = -1;

		public:
			MODEL_FIELDS
			virtual ~leaky_rexu();
			leaky_rexu();
			leaky_rexu(double a_k);
		};
	}
}