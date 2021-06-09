#pragma once
#include "pch.h"
#include "model.h"

using aurora::models::model;
using std::vector;

namespace aurora {
	namespace models {
		class sync : public model {
		public:
			ptr<model> model_template;
			vector<ptr<model>> prepared;
			vector<ptr<model>> unrolled;

		public:
			RECURRENT_FIELDS
			virtual ~sync();
			sync(ptr<model> a_model_template);

		};
	}
}