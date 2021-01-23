#pragma once
#include "macro.h"
#include "ptr.h"
#include "ref.h"
#include "tensor.h"
#include "param.h"
#include "param_sgd.h"
#include "param_mom.h"
#include <functional>
#include <vector>

using aurora::data::ptr;
using aurora::data::ref;
using std::function;
using std::vector;
using aurora::optimization::param;
using aurora::optimization::param_sgd;
using aurora::optimization::param_mom;
using aurora::math::tensor;

namespace aurora {
	namespace modeling {
		class model {
		public:
			tensor x = 0;
			tensor y = 0;
			tensor x_grad = 0;
			tensor y_grad = 0;

		public:
			MODEL_FIELDS
			virtual ~model();
			model();
			model(vector<param*>& a_pl);
			model(vector<param_sgd*>& a_pl);
			model(vector<param_mom*>& a_pl);

		};
	}
}