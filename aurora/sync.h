#pragma once
#include "pch.h"
#include "model.h"

using aurora::models::model;
using std::vector;

namespace aurora {
	namespace models {
		class sync : public model {
		public:
			Model model_template;
			vector<Model> prepared;
			vector<Model> unrolled;

		public:
			RECURRENT_FIELDS
			virtual ~sync();
			sync(Model a_model_template);

		};
		typedef ptr<sync> Sync;
	}
}