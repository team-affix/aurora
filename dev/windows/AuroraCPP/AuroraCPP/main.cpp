#pragma once
#include "main.h"
#include "general.h"


string exportParams(vector<ptr<ptr<param>>>* paramPtrVec);
void trainTnnBpg();
void trainSyncBpg();
void trainLstmBpg();
void trainMuBpg();

int main() {
	trainMuBpg();
	return 0;
}

void trainTnnBpg() {

	//seqBpg nlr = seqBpg();
	//nlr.push_back(new biasBpg());
	//nlr.push_back(new actBpg(new actFuncLR(0.05)));

	seqBpg nlr = neuronLRBpg(0.05);

	ptr<seqBpg> s = tnnBpg({ 2, 5, 1 }, { &nlr, &nlr, &nlr });

	vector <ptr<ptr<param>>> paramPtrVec = vector <ptr<ptr<param>>>();
	s->modelWise([&paramPtrVec](model* m) { initParam(m, &paramPtrVec); });

	// set up random engine for initializing param states
	uniform_real_distribution<double> u(-1, 1);
	default_random_engine re(26);

	vector <paramSgd*> paramVec = vector <paramSgd*>();
	for (ptr<ptr<param>> s : paramPtrVec) {

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
	ptr<cType> inputs = new cType{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1}
	};
	ptr<cType> desired = new cType{
		{ 0 },
		{ 1 },
		{ 1 },
		{ 0 }
	};

	uniform_int_distribution<int> ui(0, 3);

	int epoch = 0;
	while (true) {

		ptr<cType> signals = new cType({ new cType(0) });

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
			ptr<cType> sumSignals = sum1D(signals);
			cout << sumSignals->vDouble << endl;
		}
		epoch++;
	}

	cout << endl << "---------------------- Params:" << endl << exportParams(&paramPtrVec);

	return;

}

void trainSyncBpg() {

	seqBpg nlr = neuronLRBpg(0.05);
	seqBpg* templateNN = tnnBpg({ 2, 5, 1 }, &nlr);

	vector <ptr<ptr<param>>> paramPtrVec = vector <ptr<ptr<param>>>();
	templateNN->modelWise([&paramPtrVec](model* m) { initParam(m, &paramPtrVec); });

	uniform_real_distribution<double> urd(-1, 1);
	default_random_engine re(43);

	vector<paramSgd*> params = vector<paramSgd*>();
	for (ptr<ptr<param>> pptr : paramPtrVec) {
		paramSgd* p = new paramSgd();
		p->gradient = 0;
		p->learnRate = 0.02;
		p->state = urd(re);
		*pptr = p;
		params.push_back(p);
	}

	syncBpg r = syncBpg(templateNN);
	r.prep(4);
	r.unroll(4);

	ptr<cType> inputs = new cType{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	};
	ptr<cType> desired = new cType{
		{ 0 },
		{ 1 },
		{ 1 },
		{ 0 },
	};

	r.yGrad = new cType({ 
		{ 0 }, 
		{ 0 }, 
		{ 0 }, 
		{ 0 } });

	for (int epoch = 0; epoch < 1000000; epoch++) {

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

	cout << endl << "---------------------- Params:" << endl << exportParams(&paramPtrVec);
}

void trainLstmBpg() {

	seqBpg nlr = neuronLRBpg(0.05);

	lstmBpg l1 = lstmBpg(5);
	seqBpg* inNN = tnnBpg({ 2, 5 }, &nlr);
	seqBpg* outNN = tnnBpg({ 5, 1 }, &nlr);

	vector <ptr<ptr<param>>> paramPtrVec = vector <ptr<ptr<param>>>();
	inNN->modelWise([&paramPtrVec](model* m) { initParam(m, &paramPtrVec); });
	outNN->modelWise([&paramPtrVec](model* m) { initParam(m, &paramPtrVec); });
	l1.modelWise([&paramPtrVec](model* m) { initParam(m, &paramPtrVec); });

	uniform_real_distribution<double> urd(-1, 1);
	default_random_engine re(43);

	vector<paramSgd*> params = vector<paramSgd*>();
	for (ptr<ptr<param>> pptr : paramPtrVec) {
		paramSgd* p = new paramSgd();
		p->gradient = 0;
		p->learnRate = 0.02;
		p->state = urd(re);
		*pptr = p;
		params.push_back(p);
	}



	ptr<cType> inputs = new cType{ 
		{
			{0, 0},
			{0, 1},
			{1, 0},
			{1, 1},
		},
		{
			{0, 0},
			{1, 1},
			{1, 0},
			{1, 1}
		}
	};
	ptr<cType> desired = new cType{
		{
			{ 0 },
			{ 1 },
			{ 1 },
			{ 0 },
		},
		{
			{ 0 },
			{ 1 },
			{ 1 },
			{ 1 },
		}
	};

	l1.prep(4);
	l1.unroll(4);
	syncBpg rIn = syncBpg(inNN);
	rIn.prep(4);
	rIn.unroll(4);
	syncBpg rOut = syncBpg(outNN);
	rOut.prep(4);
	rOut.unroll(4);

	rOut.yGrad = make2D(4, 1);

	for (int epoch = 0; epoch < 100000; epoch++) {

		for (int i = 0; i < inputs->vVector.size(); i++) {

			rIn.x = inputs->vVector.at(i);
			rIn.fwd();
			l1.x = rIn.y;
			l1.fwd();
			rOut.x = l1.y;
			rOut.fwd();
			sub2D(rOut.y, desired->vVector.at(i), rOut.yGrad);
			rOut.bwd();
			l1.yGrad = rOut.xGrad;
			l1.bwd();
			rIn.yGrad = l1.xGrad;
			rIn.bwd();

		}

		for (paramSgd* p : params) {

			p->state -= p->learnRate * p->gradient;
			p->gradient = 0;

		}

		if (epoch % 1000 == 0) {

			cout << sum1D(sum2D(abs2D(rOut.yGrad)))->vDouble << endl;

		}
	}

	cout << endl << "---------------------- Params:" << endl << exportParams(&paramPtrVec);

}

void trainMuBpg() {

	seqBpg nlr = neuronLRBpg(0.05);
	seqBpg nth = neuronThBpg();

	int xUnits = 2;
	int cTUnits = 7;
	int hTUnits = 1;

	muBpg m1 = muBpg(xUnits, cTUnits, hTUnits);

	vector <ptr<ptr<param>>> paramPtrVec = vector <ptr<ptr<param>>>();
	m1.modelWise([&paramPtrVec](model* m) { initParam(m, &paramPtrVec); });

	uniform_real_distribution<double> urd(-0.1, 0.1);
	default_random_engine re(43);

	vector<paramMom*> params = vector<paramMom*>();
	for (ptr<ptr<param>> pptr : paramPtrVec) {
		paramMom* p = new paramMom();
		p->gradient = 0;
		p->learnRate = 0.02;
		p->state = urd(re);
		p->beta = 0.9;
		p->momentum = 0;
		*pptr = p;
		params.push_back(p);
	}



	ptr<cType> inputs = new cType{
		{
			{0, 0},
			{0, 1},
			{1, 0},
			{1, 1},
			{3.2, 1},
			{1, 1.7},
			{1, 1},
		},
		{
			{0, 0},
			{1, 1},
			{1, 0},
			{1, 1},
			{3.2, 1},
			{1, 1.7},
			{1, 1},
		}
	};
	ptr<cType> desired = new cType{
		{
			{ 0 },
			{ 1 },
			{ 1 },
			{ 0 },
			{ 0 },
			{ 0 },
			{ 0 },
		},
		{
			{ 0 },
			{ 1 },
			{ 1 },
			{ 0.25 },
			{ 0.754 },
			{ -1 },
			{ 1 },
		}
	};

	m1.prep(7);
	m1.unroll(7);

	m1.yGrad = make2D(7, 1);

	for (int epoch = 0; epoch < 10000000; epoch++) {

		for (int i = 0; i < inputs->vVector.size(); i++) {

			m1.x = inputs->vVector.at(i);
 			m1.fwd();
			sub2D(m1.y, desired->vVector.at(i), m1.yGrad);
			m1.bwd();

		}

		for (paramMom* p : params) {

			p->momentum = (p->beta * p->momentum) + (1 - p->beta) * p->gradient;
			p->state -= p->learnRate * p->momentum;
			p->gradient = 0;

		}

		if (epoch % 1000 == 0) {

			cout << sum1D(sum2D(abs2D(m1.yGrad)))->vDouble << endl;

		}
	}

	cout << endl << "---------------------- Params:" << endl << exportParams(&paramPtrVec);

}

string exportParams(vector<ptr<ptr<param>>>* paramPtrVec) {
	string result = "";
	for (int i = 0; i < paramPtrVec->size(); i++) {
		result += to_string((*paramPtrVec->at(i))->state);
		result += "\n";
	}
	return result;
}