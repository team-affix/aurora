#pragma once
#include "pch.h"
#include "model.h"

using aurora::models::model;
using std::vector;

namespace aurora {
	namespace models {
		class layer : public model {
		public:
			vector<ptr<model>> models;

		public:
			MODEL_FIELDS
			virtual ~layer();
			layer();
			layer(size_t a_a, ptr<model> a_model_template, function<void(ptr<param>&)> a_init);
			layer(size_t a_a, ptr<model> a_model_template);
			layer(initializer_list<ptr<model>> a_il);

		};
	}
}