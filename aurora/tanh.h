#pragma once
#include "pch.h"
#include "model.h"

using aurora::models::model;

namespace aurora {
	namespace models {
		class tanh : public model {
		public:
			MODEL_FIELDS
			virtual ~tanh();
			tanh();

		};
	}
}