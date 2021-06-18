#pragma once
#include "pch.h"
#include "param.h"

using aurora::params::Param;

namespace aurora {
	namespace params {
		class param_vector : public vector<Param> {
		public:
			void update();
		};
	}
}

