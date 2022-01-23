#pragma once
#include "affix-base/pch.h"
#include "model.h"

namespace aurora {
	namespace models {
		class sigmoid : public model {
		public:
			MODEL_FIELDS
			virtual ~sigmoid();
			sigmoid();

		};
		typedef affix_base::data::ptr<sigmoid> Sigmoid;
	}
}
