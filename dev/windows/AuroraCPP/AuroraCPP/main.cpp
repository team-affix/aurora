#pragma once
#include "main.h"
#include "general.h"

int main() {

	seqBpg nlr = seqBpg();
	nlr.push_back(new biasBpg());
	nlr.push_back(new actBpg(new actFuncLR(0.05)));

	seqBpg s = tnnBpg({ 2, 5, 1 }, { &nlr, &nlr, &nlr });

	vector <sPtr<sPtr<param>>> paramPtrVec = vector <sPtr<sPtr<param>>>();
	s.modelWise([&paramPtrVec](model* m) { initParam(m, &paramPtrVec); });

	// set up random engine for initializing param states
	uniform_real_distribution<double> u(-1, 1);
	default_random_engine re(26);

	vector <paramSgd*> paramVec = vector <paramSgd*>();
	for (sPtr<sPtr<param>> s : paramPtrVec) {

		// initialize type of parameter used
		paramSgd* ps = new paramSgd();

		// initialize parameter state
		ps->state = u(re);
		ps->learnRate = 0.02;
		ps->gradient = 0;

		// cause the model's parameter ptr to point to the dynamically allocated paramSgd
		*s = ps;

		paramVec.push_back(ps);

	}
	sPtr<cType> inputs = new cType{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1}
	};
	sPtr<cType> desired = new cType{
		{ 0 },
		{ 1 },
		{ 1 },
		{ 0 }
	};

	uniform_int_distribution<int> ui(0, 3);

	int epoch = 0;
	while (true) {

		sPtr<cType> signals = new cType({ new cType(0) });

		for (int i = 0; i < inputs->vVector.size(); i++) {
			int tsIndex = ui(re);
			s.x = inputs->vVector.at(tsIndex);
			s.fwd();
			s.yGrad = new cType({ new cType(0) });
			sub1D(s.y, desired->vVector.at(tsIndex), s.yGrad);
			s.bwd();
			add1D(signals, abs1D(s.yGrad), signals);
		}
		for (paramSgd* p : paramVec) {
			p->state -= p->learnRate * p->gradient;
			p->gradient = 0;
		}

		if (epoch % 1000 == 0) {
			sPtr<cType> sumSignals = sum1D(signals);
			cout << sumSignals->vDouble << endl;
		}
		epoch++;
	}

	return 0;

}