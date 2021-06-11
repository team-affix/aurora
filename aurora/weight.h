#pragma once
#include "pch.h"
#include "model.h"
#include "param.h"

using aurora::models::model;

namespace aurora {
	namespace models {
		class weight : public model {
		public:
			ptr<param> pmt = new param();

		public:
			MODEL_FIELDS
			virtual ~weight();
			weight();
			weight(function<void(ptr<param>&)> a_func);

		};
		typedef ptr<weight> Weight;
	}
}