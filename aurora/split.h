#pragma once
#include "model.h"

namespace aurora
{
	namespace models
	{
		class split : public model
		{
		public:
			std::vector<size_t> m_units;

		public:
			MODEL_FIELDS
			virtual ~split(

			);
			split(

			);
			split(
				const std::vector<size_t>& a_units
			);

		};
		typedef affix_base::data::ptr<split> Split;
	}
}
