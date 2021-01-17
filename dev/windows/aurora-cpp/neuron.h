#pragma once
#include "model.h"
#include "sequential.h"
#include <vector>

using aurora::modeling::model;
using aurora::modeling::sequential;
using std::vector;

namespace aurora {
	namespace pseudo {
		sequential* nsm();
		sequential* nth();
		sequential* nlr(double a_m);
	}
}