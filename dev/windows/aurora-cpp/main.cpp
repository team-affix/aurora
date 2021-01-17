#pragma once
#include "data.h"
#include "math.h"
#include "modeling.h"
#include "optimization.h"
#include "pseudo.h"

using namespace aurora;
using namespace aurora::data;
using namespace aurora::math;
using namespace aurora::modeling;
using namespace aurora::optimization;

int main() {

	vector<param_sgd*> pl = vector<param_sgd*>();

	ptr<sequential> s = pseudo::tnn({ 2, 5, 1 }, pseudo::nth(), pl);

	return 0;
}