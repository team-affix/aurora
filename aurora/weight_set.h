#pragma once
#include "affix-base/pch.h"
#include "model.h"
#include "weight.h"

namespace aurora {
	namespace models {
		class weight_set : public model {
		public:
			size_t m_a;
			std::vector<Weight> m_weights;

		public:
			MODEL_FIELDS
			virtual ~weight_set();
			weight_set();
			weight_set(
				size_t a_a
			);

		};
		typedef affix_base::data::ptr<weight_set> Weight_set;
	}
}
