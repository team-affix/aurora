#pragma once
#include "model.h"
#include <vector>
#include <initializer_list>

using aurora::modeling::model;
using std::vector;

namespace aurora {
	namespace modeling {
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