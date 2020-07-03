#pragma once
#include "optimization.h"

#pragma region opt
/*
void opt::initModel(model* m, vector<ptr<ptr<param>>>* paramPtrVec, uniform_real_distribution<double> u, default_random_engine re) {

	for (int i = 0; i < paramPtrVec->size(); i++) {

		ptr<ptr<param>> Ptr = paramPtrVec->at(i);

		// initialize type of parameter used
		param* p = new param();

		// initialize parameter state
		p->state = u(re);
		p->learnRate = 0.02;

		// cause the model's parameter ptr to point to the dynamically allocated paramSgd
		*Ptr = p;

	}
}
*/
#pragma endregion
