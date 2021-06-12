#pragma once
#include "pch.h"
#include "model.h"
#include "param.h"

using aurora::models::model;

namespace aurora {
	namespace models {
		class weight : public model {
		public:
			Param pmt = new param();

		public:
			MODEL_FIELDS
			virtual ~weight();
			weight();

		};
		typedef ptr<weight> Weight;
	}
}