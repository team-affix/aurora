#pragma once
#include "pch.h"

using std::vector;

namespace affix_base {
	namespace sorting {
		template<class T>
		size_t ltg_insertion_index(vector<T> a_vec, T a_val) {
			for (int i = 0; i < a_vec.size(); i++)
				if (a_vec[i] > a_val)
					return i;
			return a_vec.size();
		}
		template<class T>
		size_t gtl_insertion_index(vector<T> a_vec, T a_val) {
			for (int i = 0; i < a_vec.size(); i++)
				if (a_vec[i] < a_val)
					return i;
			return a_vec.size();
		}
	}
}