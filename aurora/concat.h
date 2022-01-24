#pragma once
#include "affix-base/pch.h"
#include "models.h"

namespace aurora
{
	namespace models
	{
		class concat : public model
		{
		public:
			std::vector<size_t> m_units;

		public:
			MODEL_FIELDS
			virtual ~concat(

			);
			concat(

			);
			concat(
				const std::vector<size_t>& a_units
			);

		};
		typedef affix_base::data::ptr<concat> Concat;
	}
}
