#pragma once
#include "model.h"
#include "layer.h"

namespace aurora
{
	namespace models
	{
		class add_1d : public model
		{
		public:
			size_t m_units = 0;

		public:
			Layer m_layer;

		public:
			MODEL_FIELDS
			virtual ~add_1d();
			add_1d();
			add_1d(
				const size_t& a_units
			);

		};
	}
}
