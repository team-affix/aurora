#pragma once
#include "pch.h"
#include "pseudo.h"
#include "basic_model.h"
#include "sequential.h"

using aurora::basic::basic_model;
using aurora::models::Sequential;
using std::vector;

namespace aurora {
	namespace basic {
		basic_model tnn(vector<size_t> a_dims);
	}
}
