#pragma once
#include "ptr.h"

namespace aurora {
	namespace optimization {
		class param {
		public:
			double state = 0;
			double learn_rate = 0;
		};
	}
}