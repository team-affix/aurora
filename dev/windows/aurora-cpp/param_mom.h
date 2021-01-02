#pragma once
#include "param_sgd.h"

namespace aurora {
	namespace optimization {
		class param_mom : public param_sgd {
		public:
			ptr<double> momentum = new double(0);
			ptr<double> beta = new double(0);
		};
	}
}