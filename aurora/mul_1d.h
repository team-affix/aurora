#pragma once
#include "affix-base/pch.h"
#include "model.h"
#include "layer.h"

namespace aurora
{
	namespace models
	{
		class mul_1d : public model
		{
		public:
			size_t m_units = 0;

		public:
			Layer m_layer;

		public:
			MODEL_FIELDS
			virtual ~mul_1d();
			mul_1d(
				const size_t& a_units
			);

		};
		typedef affix_base::data::ptr<mul_1d> Mul_1d;
	}
}
