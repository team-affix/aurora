#pragma once
#include "model.h"

using aurora::modeling::model;

namespace aurora {
	namespace modeling {
		class tanh : public model {
		public:
			MODEL_FIELDS
			virtual ~tanh();
			tanh();

		};
	}
}