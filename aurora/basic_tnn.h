#pragma once
#include "pch.h"
#include "pseudo.h"
#include "sequential.h"
#include "param_vector.h"

using aurora::models::Sequential;
using std::vector;

namespace aurora {
	namespace basic {
		Model tnn(vector<size_t> a_dims, param_vector& a_param_vec);
	}
}
