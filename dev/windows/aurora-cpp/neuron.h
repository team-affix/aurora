#pragma once
#include "model.h"
#include "sequential.h"
#include <vector>

using aurora::models::model;
using aurora::models::sequential;
using std::vector;

namespace aurora {
	namespace pseudo {
		sequential* nsm();
		sequential* nth();
		sequential* nlr(double a_m);
	}
}