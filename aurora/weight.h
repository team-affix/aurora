#pragma once
#include "affix-base/pch.h"
#include "model.h"
#include "param.h"

namespace aurora {
	namespace models {
		class weight : public model {
		public:
			aurora::params::Param pmt = new aurora::params::param();

		public:
			MODEL_FIELDS
			virtual ~weight();
			weight();

		};
		typedef affix_base::data::ptr<weight> Weight;
	}
}
