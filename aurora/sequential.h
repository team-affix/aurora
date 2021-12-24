#pragma once
#include "affix-base/pch.h"
#include "model.h"
#include "param.h"
#include "param_sgd.h"
#include "param_mom.h"

namespace aurora {
	namespace models {
		class sequential : public model {
		public:
			vector<Model> models;

		public:
			MODEL_FIELDS
			virtual ~sequential();
			sequential();
			sequential(std::initializer_list<Model> a_models);
			sequential(vector<Model> a_models);

		};
		typedef affix_base::data::ptr<sequential> Sequential;
	}
}
