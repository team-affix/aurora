#pragma once
#include "pch.h"
#include "model.h"
#include "recurrent.h"

using aurora::models::model;
using aurora::models::recurrent;
using std::vector;

namespace aurora {
	namespace models {
		class sync : public recurrent {
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