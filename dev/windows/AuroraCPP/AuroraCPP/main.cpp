#pragma once
#include "main.h"
#include "general.h"


int main() {

	seqBpg nlr = seqBpg();
	nlr.push_back(new biasBpg());
	nlr.push_back(new actBpg(new actFuncLR(0.05)));

	seqBpg s = seqBpg();
	s.push_back(new layerBpg(2, &nlr));
	s.push_back(new wJuncBpg(2, 5));
	s.push_back(new layerBpg(5, &nlr));
	s.push_back(new wJuncBpg(5, 1));
	s.push_back(new layerBpg(1, &nlr));

	vector <sPtr<sPtr<param>>> paramVec = vector <sPtr<sPtr<param>>>();
	s.modelWise([&paramVec] (model* m) { initParam(m, &paramVec); });

	uniform_real_distribution<double> u(6);
	default_random_engine(3);

	for (sPtr<sPtr<param>> s : paramVec) {

		// initialize type of parameter used
		paramSgd* ps = new paramSgd();

		// initialize parameter values
		ps->state = 0.2;
		ps->learnRate = 0;
		ps->gradient = 0;

		// cause the model's parameter ptr to point to the dynamically allocated paramSgd
		*s = ps;

	}

	s.x = new cType({1, 2});
	s.fwd();

	return 0;

}