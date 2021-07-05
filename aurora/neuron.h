#pragma once
#include "pch.h"
#include "model.h"
#include "sequential.h"

using aurora::models::model;
using aurora::models::Sequential;
using std::vector;

namespace aurora {
	namespace pseudo {
		Sequential nsm();
		Sequential nth();
		Sequential nth(double a_a, double a_b, double a_c);
		Sequential nlr(double a_m);
		Sequential nlrexu(double a_k);
	}
}