#pragma once
#include "pch.h"
#include "model.h"

using aurora::models::model;

namespace aurora {
	namespace models {
		class bias : public model {
		public:
			Param pmt = new param();

		public:
			MODEL_FIELDS
			virtual ~bias();
			bias();

		};
		typedef ptr<bias> Bias;
	}
}