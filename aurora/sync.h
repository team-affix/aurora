#pragma once
#include "affix-base/pch.h"
#include "model.h"
#include "recurrent.h"

namespace aurora {
	namespace models {
		class sync : public recurrent {
		public:
			Model model_template;
			std::vector<Model> prepared;
			std::vector<Model> unrolled;

		public:
			RECURRENT_FIELDS
			virtual ~sync();
			sync(Model a_model_template);

		};
		typedef affix_base::data::ptr<sync> Sync;
	}
}
