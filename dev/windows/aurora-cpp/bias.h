#pragma once
#include "model.h"

using aurora::modeling::model;

namespace aurora {
	namespace modeling {
		class bias : public model {
		public:
			ptr<param> pmt = new param();

		public:
			MODEL_FIELDS
			virtual ~bias();
			bias();
			bias(function<void(ptr<param>&)> a_init);

		};
	}
}