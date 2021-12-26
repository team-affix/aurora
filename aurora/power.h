#pragma once
#include "affix-base/pch.h"
#include "model.h"

namespace aurora {
	namespace models {
		class power : public model {
		public:
			size_t m_units = 0;

		public:
			aurora::maths::tensor m_amount;
			aurora::maths::tensor m_amount_grad;

		public:
			MODEL_FIELDS
			virtual ~power();
			power();
			power(
				size_t a_units
			);

		};
		typedef affix_base::data::ptr<power> Power;
	}
}
