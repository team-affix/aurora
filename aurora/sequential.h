#pragma once
#include "model.h"
#include "param.h"
#include "param_sgd.h"
#include "param_mom.h"
#include <vector>
#include <initializer_list>

using aurora::models::model;
using std::vector;
using std::initializer_list;
using std::back_inserter;

namespace aurora {
	namespace models {
		class sequential : public model {
		public:
			vector<ptr<model>> models;

		public:
			MODEL_FIELDS
			virtual ~sequential();
			sequential();
			sequential(initializer_list<ptr<model>> a_il);
			sequential(vector<ptr<model>> a_models);

		};
	}
}