#pragma once
#include "affix-base/pch.h"
#include "model.h"

namespace aurora {
	namespace models {
		class tanh : public model {
		public:
			double m_a = 1;
			double m_b = 1;
			double m_c = 0;

		public:
			MODEL_FIELDS
			virtual ~tanh();
			tanh();
			tanh(
				double a_a,
				double a_b,
				double a_c
			);

		};
		typedef affix_base::data::ptr<tanh> Tanh;
	}
}
