#pragma once
#include "model.h"

using aurora::modeling::model;

namespace aurora {
	namespace modeling {
		class weight : public model {
		public:
			param* pmt = new param();

		public:
			virtual ~weight();
			weight();
			weight(vector<param*>& pl);
			weight(vector<param_sgd*>& pl);
			weight(vector<param_mom*>& pl);

		public:
			virtual void fwd();
			virtual void bwd();
			virtual void recur(function<void(model&)> func);
			virtual void compile();

		};
	}
}