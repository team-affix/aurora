#pragma once
#include "param_sgd.h"

namespace aurora {
	namespace optimization {
		class param_mom : public param_sgd {
		public:
			double momentum = 0;
			double beta = 0;
		};
	}
}