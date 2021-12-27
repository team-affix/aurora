#pragma once
#include "model.h"
#include "mul_1d.h"
#include "sum_1d.h"

namespace aurora
{
	namespace models
	{
		class dot_1d : public model
		{
		public:
			size_t m_units = 0;

		public:
			Mul_1d m_mul_1d;
			Sum_1d m_sum_1d;

		public:
			MODEL_FIELDS
			virtual ~dot_1d();
			dot_1d(
				const size_t& a_units
			);

		};
		typedef affix_base::data::ptr<dot_1d> Dot_1d;
	}
}
