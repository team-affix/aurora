#pragma once
#include "affix-base/pch.h"
#include "model.h"
#include "recurrent.h"

namespace aurora {
	namespace models {
		class sync : public recurrent {
		public:
			Model m_model_template;
			std::vector<Model> m_prepared;
			std::vector<Model> m_unrolled;

		public:
			RECURRENT_FIELDS
			virtual ~sync();
			sync(
				Model a_model_template
			);

		};
		typedef affix_base::data::ptr<sync> Sync;
	}
}
