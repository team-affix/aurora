#pragma once
#include "affix-base/pch.h"
#include "model.h"

namespace aurora
{
	namespace models
	{
		class sum_1d : public model
		{
		public:
			size_t m_units = 0;

		public:
			MODEL_FIELDS
			virtual ~sum_1d();
			sum_1d(
				const size_t& a_units
			);

		};
		typedef affix_base::data::ptr<sum_1d> Sum_1d;
	}
}
