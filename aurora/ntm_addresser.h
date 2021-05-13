#pragma once
#include "model.h"
#include "ntm_rh.h"

using aurora::models::model;

namespace aurora {
	namespace models {
		class ntm_addresser : public model {
		public:
			tensor m;

		public:
			MODEL_FIELDS
			virtual ~ntm_addresser();
			ntm_addresser();
			/*ntm_addresser(size_t a_)*/

		};
	}
}
