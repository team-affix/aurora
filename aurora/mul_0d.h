#pragma once
#include "affix-base/pch.h"
#include "model.h"

namespace aurora
{
	namespace models
	{
		class mul_0d : public model
		{
		public:
			MODEL_FIELDS
			virtual ~mul_0d();
			mul_0d();

		};
		typedef affix_base::data::ptr<mul_0d> Mul_0d;
	}
}
