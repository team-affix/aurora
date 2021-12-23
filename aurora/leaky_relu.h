#pragma once
#include "affix-base/pch.h"
#include "model.h"

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
		typedef ptr<leaky_relu> Leaky_relu;
	}
}