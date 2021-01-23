#pragma once
#include "model.h"
#include "param.h"

using aurora::modeling::model;

namespace aurora {
	namespace modeling {
		class weight : public model {
		public:
			ptr<param> pmt = new param();

		public:
			MODEL_FIELDS
			virtual ~weight();
			weight();
			weight(vector<param*>& a_pl);
			weight(vector<param_sgd*>& a_pl);
			weight(vector<param_mom*>& a_pl);

		};
	}
}