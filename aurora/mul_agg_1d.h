#pragma once
#include "affix-base/pch.h"
#include "model.h"

namespace aurora
{
	namespace models
	{
		class mul_agg_1d : public model
		{
		public:
			size_t m_units = 0;

		public:
			MODEL_FIELDS
			virtual ~mul_agg_1d();
			mul_agg_1d();
			mul_agg_1d(
				const size_t& a_units
			);

		};
	}
}
