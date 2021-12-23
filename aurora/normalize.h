#pragma once
#include "affix-base/pch.h"
#include "model.h"

namespace aurora {
	namespace models {
		class normalize : public model {
		public:
			size_t units = 0;

		public:
			tensor x_abs;

		public:
			double sum = 0;

		public:
			MODEL_FIELDS
			virtual ~normalize();
			normalize();
			normalize(size_t a_units);

		};
		typedef ptr<normalize> Normalize;
	}
}