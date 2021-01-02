#pragma once
#include "ptr.h"

namespace aurora {
	namespace optimization {
		class param {
		public:
			ptr<double> state = new double(0);
			ptr<double> learn_rate = new double(0);
		};
	}
}