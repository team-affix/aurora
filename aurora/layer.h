#pragma once
#include "pch.h"
#include "model.h"

using aurora::models::model;
using std::vector;

namespace aurora {
	namespace models {
		class layer : public model {
		public:
			vector<Model> models;

		public:
			MODEL_FIELDS
			virtual ~layer();
			layer();
			layer(size_t a_height, Model a_model_template);
			layer(initializer_list<Model> a_models);
			layer(vector<Model> a_models);

		};
		typedef ptr<layer> Layer;
	}
}