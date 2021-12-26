#pragma once
#include "affix-base/pch.h"
#include "model.h"

namespace aurora {
	namespace models {
		class cos_sim : public model {
		public:
			size_t m_units = 0;

		public:
			double m_magnitude_0 = 0;
			double m_magnitude_1 = 0;
			double m_magnitude_product = 0;
			double m_dot_product = 0;

		public:
			MODEL_FIELDS
			virtual ~cos_sim();
			cos_sim();
			cos_sim(size_t a_units);

		};
		typedef affix_base::data::ptr<cos_sim> Cos_sim;
	}
}
