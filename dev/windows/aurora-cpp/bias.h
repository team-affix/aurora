#pragma once
#include "model.h"
#include "param.h"
#include "param_sgd.h"
#include "param_mom.h"
#include <vector>

using aurora::modeling::model;
using std::vector;

namespace aurora {
	namespace modeling {
		class bias : public model {
		public:
			param* pmt = new param();

		public:
			virtual ~bias();
			bias();
			bias(vector<param*>& pl);
			bias(vector<param_sgd*>& pl);
			bias(vector<param_mom*>& pl);

		public:
			virtual void fwd();
			virtual void bwd();
			virtual void recur(function<void(model&)> func);
			virtual void compile();

		};
	}
}