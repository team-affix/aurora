#pragma once
#include "pch.h"
#include "model.h"
#include "param.h"

using aurora::models::Model;
using aurora::params::Param;

namespace aurora {
	namespace basic {
		class basic_model {
		public:
			Model m_model;
			vector<param*> m_params;

		public:
			void update();

		};
	}
}
