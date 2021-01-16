#pragma once
#include "param.h"

using aurora::optimization::param;

namespace aurora {
	namespace optimization {
		class param_sgd : public param {
		public:
			ptr<double> gradient_ptr = new double(0);

		public:
			virtual ~param_sgd();

		public:
			double& gradient();

		public:
			virtual param* clone();
			virtual param* to_param();
			virtual param_sgd* to_param_sgd();
			virtual param_mom* to_param_mom();

		};
	}
}