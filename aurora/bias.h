#pragma once
#include "affix-base/pch.h"
#include "model.h"

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