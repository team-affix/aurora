#pragma once
#include "model.h"

using aurora::modeling::model;

namespace aurora {
	namespace modeling {
		class sigmoid : public model {
		public:
			MODEL_FIELDS
			virtual ~sigmoid();
			sigmoid();

		};
	}
}