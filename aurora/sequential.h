#pragma once
#include "pch.h"
#include "model.h"
#include "param.h"
#include "param_sgd.h"
#include "param_mom.h"

using aurora::models::model;
using std::vector;
using std::initializer_list;
using std::back_inserter;

namespace aurora {
	namespace models {
		class sequential : public model {
		public:
			vector<Model> models;

		public:
			MODEL_FIELDS
			virtual ~sequential();
			sequential();
			sequential(initializer_list<Model> a_il);
			sequential(vector<Model> a_models);

		};
		typedef ptr<sequential> Sequential;
	}
}