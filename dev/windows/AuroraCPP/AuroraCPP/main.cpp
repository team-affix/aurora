#pragma once
#include "main.h"
#include "general.h"

void trainTnnBpg();
void trainRctBpg();

int main() {
	trainRctBpg();
	return 0;
}

void trainTnnBpg() {

	//seqBpg nlr = seqBpg();
	//nlr.push_back(new biasBpg());
	//nlr.push_back(new actBpg(new actFuncLR(0.05)));

	seqBpg nlr = neuronLRBpg(0.05);

	sPtr<seqBpg> s = tnnBpg({ 2, 5, 1 }, { &nlr, &nlr, &nlr });

	vector <sPtr<sPtr<param>>> paramPtrVec = vector <sPtr<sPtr<param>>>();
	s->modelWise([&paramPtrVec](model* m) { initParam(m, &paramPtrVec); });

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
			s->x = inputs->vVector.at(tsIndex);
			s->fwd();
			s->yGrad = new cType({ new cType(0) });
			sub1D(s->y, desired->vVector.at(tsIndex), s->yGrad);
			s->bwd();
			add1D(signals, abs1D(s->yGrad), signals);
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

	return;

}

void trainRctBpg() {

	seqBpg nlr = neuronLRBpg(0.05);
	seqBpg* templateNN = tnnBpg({ 2, 5, 1 }, { &nlr, &nlr, &nlr });

	vector <sPtr<sPtr<param>>> paramPtrVec = vector <sPtr<sPtr<param>>>();
	templateNN->modelWise([&paramPtrVec](model* m) { initParam(m, &paramPtrVec); });

	uniform_real_distribution<double> urd(-1, 1);
	default_random_engine re(43);

	vector<paramSgd*> params = vector<paramSgd*>();
	for (sPtr<sPtr<param>> pptr : paramPtrVec) {
		paramSgd* p = new paramSgd();
		p->gradient = 0;
		p->learnRate = 0.02;
		p->state = urd(re);
		*pptr = p;
		params.push_back(p);
	}

	rctBpg r = rctBpg(templateNN);
	r.prep(4);
	r.unroll(4);

	sPtr<cType> inputs = new cType{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	};
	sPtr<cType> desired = new cType{
		{ 0 },
		{ 1 },
		{ 1 },
		{ 0 },
	};

	r.yGrad = new cType({ { 0 }, { 0 }, { 0 }, { 0 } });
	for (int epoch = 0; epoch < 100000; epoch++) {

		r.x = inputs;
		r.fwd();
		sub2D(r.y, desired, r.yGrad);
		r.bwd();

		for (paramSgd* p : params) {
			p->state -= p->learnRate * p->gradient;
			p->gradient = 0;
		}

		if (epoch % 1000 == 0) {
			cout << sum1D(sum2D(abs2D(r.yGrad)))->vDouble << endl;
		}
	}
}