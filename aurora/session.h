#pragma once
#include "pch.h"
#include "basic_model.h"

using aurora::basic::basic_model;
using aurora::params::param;
using aurora::models::Model;
using std::vector;

namespace aurora {
	namespace basic {
		class session {
		protected:
			basic_model m_model_info;
			
		};
	}
}