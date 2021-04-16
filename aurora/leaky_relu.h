#pragma once
#include "pch.h"
#include "model.h"

using aurora::models::model;

namespace aurora {
	namespace models {
		class leaky_relu : public model {
		public:
			ptr<double> m = new double(0);

		public:
			MODEL_FIELDS
			virtual ~leaky_relu();
			leaky_relu(double a_m);

		};
	}
}