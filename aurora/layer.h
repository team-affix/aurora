#pragma once
#include "affix-base/pch.h"
#include "model.h"

namespace aurora {
	namespace models {
		class layer : public model {
		public:
			std::vector<Model> models;

		public:
			MODEL_FIELDS
			virtual ~layer();
			layer();
			layer(size_t a_height, Model a_model_template);
			layer(std::initializer_list<Model> a_models);
			layer(std::vector<Model> a_models);

		};
		typedef affix_base::data::ptr<layer> Layer;
	}
}
