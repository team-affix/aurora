#pragma once
#include "affix-base/pch.h"
#include "model.h"

namespace aurora {
	namespace models {
		class interpolate : public model {
		public:
			size_t m_units = 0;

		public:
			double m_amount_compliment = 0;

		public:
			aurora::maths::tensor m_amount;
			aurora::maths::tensor m_amount_grad;

		public:
			MODEL_FIELDS
			virtual ~interpolate();
			interpolate();
			interpolate(
				size_t a_units
			);

		};
		typedef affix_base::data::ptr<interpolate> Interpolate;
	}
}
