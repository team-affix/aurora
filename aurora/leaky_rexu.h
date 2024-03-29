#pragma once
#include "affix-base/pch.h"
#include "model.h"

namespace aurora {
	namespace models {
		class leaky_rexu : public model {
		public:
			double m_k = 0;
			double m_k_minus_one = -1;

		public:
			MODEL_FIELDS
			virtual ~leaky_rexu();
			leaky_rexu();
			leaky_rexu(
				double a_k
			);

		};
		typedef affix_base::data::ptr<leaky_rexu> Leaky_rexu;
	}
}
