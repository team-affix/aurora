#pragma once
#include "affix-base/pch.h"
#include "model.h"

namespace aurora {
	namespace models {
		class bias : public model {
		public:
			aurora::params::Param m_param = new aurora::params::param();

		public:
			MODEL_FIELDS
			virtual ~bias();
			bias();

		};
		typedef affix_base::data::ptr<bias> Bias;
	}
}
