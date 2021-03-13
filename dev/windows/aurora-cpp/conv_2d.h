#pragma once
#include "model.h"
#include "sync.h"

using aurora::models::model;
using aurora::models::sync;

namespace aurora {
	namespace models {
		class conv_2d : public model {
		public:
			size_t a;
			size_t b;
			size_t stride_len;
			//ptr<sync> ;

		public:
			//MODEL_FIELDS
			//conv_2d();
			//conv_2d(size_t a_a, size_t a_b);
			//conv_2d(size_t a_a, size_t a_b, size_t a_stride_len);

		};
	}
}