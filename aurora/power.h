#pragma once
#include "affix-base/pch.h"
#include "model.h"

namespace aurora {
	namespace models {
		class power : public model {
		public:
			size_t units = 0;

		public:
			aurora::maths::tensor amount;
			aurora::maths::tensor amount_grad;

		public:
			MODEL_FIELDS
			virtual ~power();
			power();
			power(size_t a_units);

		};
		typedef affix_base::data::ptr<power> Power;
	}
}
