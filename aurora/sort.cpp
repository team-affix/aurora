#include "pch.h"
#include "sort.h"

using namespace aurora;

size_t pseudo::ltg_insertion_index(vector<double> a_vec, double a_val) {
	for (int i = 0; i < a_vec.size(); i++)
		if (a_vec[i] > a_val)
			return i;
	return a_vec.size();
}

size_t pseudo::gtl_insertion_index(vector<double> a_vec, double a_val) {
	for (int i = 0; i < a_vec.size(); i++)
		if (a_vec[i] < a_val)
			return i;
	return a_vec.size();
}