#pragma once
#include "param.h"

namespace aurora {
	namespace optimization {
		class param_sgd : public param {
		public:
			double gradient = 0;
		};
	}
}