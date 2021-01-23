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
			bias(vector<param*>& a_pl);
			bias(vector<param_sgd*>& a_pl);
			bias(vector<param_mom*>& a_pl);

		};
	}
}