#pragma once
#include "model.h"
#include "layer.h"
#include "sum_1d.h"

namespace aurora
{
	namespace models
	{
		class parameterized_dot_1d : public model
		{
		public:
			size_t m_units;

		public:
			Layer m_layer;
			Sum_1d m_sum_1d;

		public:
			MODEL_FIELDS
			virtual ~parameterized_dot_1d();
			parameterized_dot_1d(
				const size_t& a_units
			);

		};
		typedef affix_base::data::ptr<parameterized_dot_1d> Parameterized_dot_1d;
	}
}
