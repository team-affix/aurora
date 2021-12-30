#pragma once
#include "model.h"

namespace aurora
{
	namespace models
	{
		class rounding_spc : public model
		{
		public:
			static std::uniform_real_distribution<double> s_urd;

		public:
			MODEL_FIELDS
			virtual ~rounding_spc();
			rounding_spc();

		};
	}
}
