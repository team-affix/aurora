#pragma once
#include "model.h"

using aurora::modeling::model;

namespace aurora {
	namespace modeling {
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