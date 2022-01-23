#pragma once
#include "affix-base/pch.h"
#include "model.h"

namespace aurora {
	namespace models {
		class normalize : public model {
		public:
			size_t m_units = 0;

		public:
			aurora::maths::tensor m_x_abs;

		public:
			double m_sum = 0;

		public:
			MODEL_FIELDS
			virtual ~normalize();
			normalize();
			normalize(
				size_t a_units
			);

		};
		typedef affix_base::data::ptr<normalize> Normalize;
	}
}
