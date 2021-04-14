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
using aurora::params::param;
using aurora::params::param_sgd;
using aurora::params::param_mom;
using aurora::math::tensor;

namespace aurora {
	namespace models {
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
			model(function<void(ptr<param>&)> a_init);

		};
	}
}