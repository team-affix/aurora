#pragma once
#include "pch.h"
#include "model.h"

using aurora::models::model;

namespace aurora {
	namespace models {
		class sigmoid : public model {
		public:
			MODEL_FIELDS
			virtual ~sigmoid();
			sigmoid();

		};
		typedef ptr<sigmoid> Sigmoid;
	}
}