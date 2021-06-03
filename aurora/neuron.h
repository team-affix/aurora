#pragma once
#include "pch.h"
#include "model.h"
#include "sequential.h"

using aurora::models::model;
using aurora::models::sequential;
using std::vector;

namespace aurora {
	namespace pseudo {
		sequential* nsm();
		sequential* nth();
		sequential* nlr(double a_m);
		sequential* nlrexu(double a_k);
	}
}