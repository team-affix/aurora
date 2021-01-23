#pragma once
#include "model.h"
#include <vector>

using aurora::modeling::model;
using std::vector;

namespace aurora {
	namespace modeling {
		class layer : public model {
		public:
			vector<ptr<model>> models;

		public:
			MODEL_FIELDS
			virtual ~layer();
			layer();
			layer(size_t a_a, ptr<model> a_model_template, vector<param*>& a_pl);
			layer(size_t a_a, ptr<model> a_model_template, vector<param_sgd*>& a_pl);
			layer(size_t a_a, ptr<model> a_model_template, vector<param_mom*>& a_pl);
			layer(initializer_list<ptr<model>> a_il);

		};
	}
}