#pragma once
#include "affix-base/pch.h"
#include "macro.h"
#include "affix-base/ptr.h"
#include "affix-base/ref.h"
#include "tensor.h"
#include "param.h"
#include "param_sgd.h"
#include "param_mom.h"

namespace aurora {
	namespace models {
		class model {
		public:
			aurora::maths::tensor x = 0;
			aurora::maths::tensor y = 0;
			aurora::maths::tensor x_grad = 0;
			aurora::maths::tensor y_grad = 0;

		public:
			MODEL_FIELDS
			virtual ~model();
			model();

		};
		typedef affix_base::data::ptr<model> Model;
	}
}
