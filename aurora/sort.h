#pragma once
#include "pch.h"

using std::vector;

namespace aurora {
	namespace pseudo {
		size_t ltg_insertion_index(vector<double> a_vec, double a_val);
		size_t gtl_insertion_index(vector<double> a_vec, double a_val);
	}
}