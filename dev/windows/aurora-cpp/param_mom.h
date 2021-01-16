#pragma once
#include "param_sgd.h"

namespace aurora {
	namespace optimization {
		class param_mom : public param_sgd {
		public:
			ptr<double> momentum_ptr = new double(0);
			ptr<double> beta_ptr = new double(0);

		public:
			virtual ~param_mom();

		public:
			double& momentum();
			double& beta();

		public:
			virtual param* clone();
			virtual param* to_param();
			virtual param_sgd* to_param_sgd();
			virtual param_mom* to_param_mom();

		};
	}
}