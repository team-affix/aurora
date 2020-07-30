#pragma once
#include "main.h"
#include "general.h"
#include <fstream>
#include <filesystem>

using namespace std;
using namespace filesystem;

string exportParams(vector<ptr<ptr<param>>>* paramPtrVec);
void trainTnnBpg();
void trainTnnMut();
void trainSyncBpg();
void trainSyncMut();
void trainLstmBpg();
void trainLstmMut();
void trainMuBpg();
void trainMuMut();
void trainAttBpg();
void trainAccelerator();
void trainDigitRecognizer();

int main() {
	trainLstmBpg();
	return 0;
}

void trainTnnBpg() {

	ptr<model> nlr = neuronLRBpg(0.3);
	ptr<seqBpg> s = tnnBpg({ 2, 5, 1 }, { nlr, nlr, nlr });

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

		ptr<cType> signals = make1D(1);

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

void trainTnnMut() {

	ptr<model> nlr = neuronLR(0.3);
	ptr<seq> s = tnn({ 2, 5, 1 }, { nlr, nlr, nlr });

	vector <ptr<ptr<param>>> paramPtrVec = vector <ptr<ptr<param>>>();
	s->modelWise([&paramPtrVec](model* m) { initParam(m, &paramPtrVec); });

	// set up random engine for initializing param states
	uniform_real_distribution<double> u(-1, 1);
	default_random_engine re(43);

	vector <param*> paramVec = vector <param*>();
	for (ptr<ptr<param>> s : paramPtrVec) {
		// initialize type of parameter used
		param* ps = new param();
		// initialize parameter state
		ps->state = u(re);
		ps->learnRate = 1;
		// cause the model's parameter ptr to point to the dynamically allocated param
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

	int genSize = 40;

	genePool g(u, re, &paramVec, genSize, 0.005);
	g.initParent();
	g.initChildren();
	g.populateParent();
	
	ptr<cType> childCost = make1D(1);
	ptr<cType> childTSCost = make1D(1);
	ptr<cType> bestChildCost = make1D(1);  bestChildCost->vVector.at(0)->vDouble = 99999999;
	int bestChildIndex;

	for (int epoch = 0; true; epoch++) {

		g.birthGeneration();

		bestChildIndex = 0;
		bestChildCost->vVector.at(0)->vDouble = 99999999;

		for (int childIndex = 0; childIndex < genSize; childIndex++) {
			clear1D(childCost);
			g.populateParams(childIndex);
			for (int tsIndex = 0; tsIndex < inputs->vVector.size(); tsIndex++) {
				ptr<cType> x = inputs->vVector.at(tsIndex);
				ptr<cType> des = desired->vVector.at(tsIndex);
				s->x = x;
				s->fwd();
				sub1D(s->y, des, childTSCost);
				abs1D(childTSCost, childTSCost);
				add1D(childCost, childTSCost, childCost);
			}

			if (childCost->vVector.at(0)->vDouble < bestChildCost->vVector.at(0)->vDouble) {
				bestChildIndex = childIndex;
				copy1D(childCost, bestChildCost);
			}
		}
		g.makeParent(bestChildIndex);

		if (epoch % 1000 == 0) {
			double bcc = bestChildCost->vVector.at(0)->vDouble;
			cout << bcc << endl;
		}

	}


	return;

}

void trainSyncBpg() {

	ptr<model> nlr = neuronLRBpg(0.3);
	seqBpg* templateNN = tnnBpg({ 2, 5, 1 }, nlr);

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

void trainSyncMut() {
	ptr<model> nlr = neuronLR(0.3);
	seq* templateNN = tnn({ 2, 5, 1 }, nlr);

	vector <ptr<ptr<param>>> paramPtrVec = vector <ptr<ptr<param>>>();
	templateNN->modelWise([&paramPtrVec](model* m) { initParam(m, &paramPtrVec); });

	uniform_real_distribution<double> urd(-1, 1);
	default_random_engine re(43);

	vector<param*> params = vector<param*>();
	for (ptr<ptr<param>> pptr : paramPtrVec) {
		param* p = new param();
		p->learnRate = 1;
		p->state = urd(re);
		*pptr = p;
		params.push_back(p);
	}

	sync s = sync(templateNN);
	s.prep(4);
	s.unroll(4);

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

	int genSize = 40;
	genePool g(urd, re, &params, genSize, 0.002);
	g.initParent();
	g.initChildren();
	g.populateParent();

	ptr<cType> childBestCost = new cType(999999999);
	ptr<cType> childCost0D = new cType();
	ptr<cType> childCost1D = make1D(1);
	ptr<cType> childCost2D = make2D(4, 1);
	int bestChildIndex;

	for (int epoch = 0; true; epoch++) {

		childBestCost->vDouble = 999999999;
		bestChildIndex = 0;

		g.birthGeneration();

		for (int childIndex = 0; childIndex < genSize; childIndex++) {

			g.populateParams(childIndex);

			s.x = inputs;
			s.index = 0;
			s.fwd();
			sub2D(s.y, desired, childCost2D);
			abs2D(childCost2D, childCost2D);
			sum2D(childCost2D, childCost1D);
			sum1D(childCost1D, childCost0D);

			if (childCost0D->vDouble < childBestCost->vDouble) {
				childBestCost->vDouble = childCost0D->vDouble;
				bestChildIndex = childIndex;
			}
		}

		g.makeParent(bestChildIndex);

		if (epoch % 10 == 0) {
			cout << childBestCost->vDouble << endl;
		}

	}
}

void trainLstmBpg() {

	ptr<model> nlr = neuronLRBpg(0.3);

	lstmBpg l1 = lstmBpg(7);
	ptr<model> inNN = tnnBpg({ 2, 7 }, nlr);
	ptr<model> outNN = tnnBpg({ 7, 1 }, nlr);

	vector <ptr<ptr<param>>> paramPtrVec = vector <ptr<ptr<param>>>();
	inNN->modelWise([&paramPtrVec](model* m) { initParam(m, &paramPtrVec); });
	outNN->modelWise([&paramPtrVec](model* m) { initParam(m, &paramPtrVec); });
	l1.modelWise([&paramPtrVec](model* m) { initParam(m, &paramPtrVec); });

	uniform_real_distribution<double> urd(-1, 1);
	default_random_engine re(46);

	vector<paramMom*> params = vector<paramMom*>();
	for (ptr<ptr<param>> pptr : paramPtrVec) {
		paramMom* p = new paramMom();
		p->gradient = 0;
		p->learnRate = 0.02;
		p->state = urd(re);
		p->momentum = 0;
		p->beta = 0.9;
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

	ptr<cType> signals = make2D(4, 1);
	ptr<cType> absYGrad = make2D(4, 1);

	for (int epoch = 0; epoch < 100000; epoch++) {

		clear2D(signals);

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

			abs2D(rOut.yGrad, absYGrad);
			add2D(signals, absYGrad, signals);
		}

		for (paramMom* p : params) {

			p->momentum = p->beta * p->momentum + (1 - p->beta) * p->gradient;
			p->state -= p->learnRate * p->momentum;
			p->gradient = 0;

		}

		if (epoch % 1000 == 0) {

			cout << sum1D(sum2D(abs2D(signals)))->vDouble << endl;

		}
	}

	cout << endl << "---------------------- Params:" << endl << exportParams(&paramPtrVec);

}

void trainLstmMut() {

	ptr<model> nlr = neuronLRBpg(0.3);

	lstmBpg l1 = lstmBpg(5);
	ptr<model> inNN = tnnBpg({ 2, 5 }, nlr);
	ptr<model> outNN = tnnBpg({ 5, 1 }, nlr);

	vector <ptr<ptr<param>>> paramPtrVec = vector <ptr<ptr<param>>>();
	inNN->modelWise([&paramPtrVec](model* m) { initParam(m, &paramPtrVec); });
	outNN->modelWise([&paramPtrVec](model* m) { initParam(m, &paramPtrVec); });
	l1.modelWise([&paramPtrVec](model* m) { initParam(m, &paramPtrVec); });

	uniform_real_distribution<double> urd(-1, 1);
	default_random_engine re(45);

	vector<param*> params = vector<param*>();
	for (ptr<ptr<param>> pptr : paramPtrVec) {
		param* p = new param();
		p->learnRate = 10;
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

	int genSize = 60;
	genePool g(urd, re, &params, genSize, 0.005);
	g.initParent();
	g.initChildren();
	g.populateParent();

	ptr<cType> childTSCost2D = make2D(4, 1);
	ptr<cType> childCost2D = make2D(4, 1);
	ptr<cType> childCost1D = make1D(1);
	ptr<cType> childCost0D = new cType();
	ptr<cType> childBestCost = new cType();
	int bestChildIndex;

	for (int epoch = 0; true; epoch++) {

		g.birthGeneration();
		childBestCost->vDouble = 9999999;
		bestChildIndex = 0;

		for (int childIndex = 0; childIndex < genSize; childIndex++) {
			clear2D(childCost2D);
			for (int tsIndex = 0; tsIndex < inputs->vVector.size(); tsIndex++) {
				g.populateParams(childIndex);
				rIn.index = l1.index = rOut.index = 0;
				rIn.x = inputs->vVector.at(tsIndex);
				rIn.fwd();
				l1.x = rIn.y;
				l1.fwd();
				rOut.x = l1.y;
				rOut.fwd();
				sub2D(rOut.y, desired->vVector.at(tsIndex), childTSCost2D);
				abs2D(childTSCost2D, childTSCost2D);
				add2D(childCost2D, childTSCost2D, childCost2D);
			}
			sum2D(childCost2D, childCost1D);
			sum1D(childCost1D, childCost0D);

			if (childCost0D->vDouble < childBestCost->vDouble) {
				childBestCost->vDouble = childCost0D->vDouble;
				bestChildIndex = childIndex;
			}

		}
		g.makeParent(bestChildIndex);

		if (epoch % 10 == 0) {
			cout << childBestCost->vDouble << endl;
		}

	}


}

void trainMuBpg() {

	muBpg m1 = muBpg(2, 7, 1);

	vector <ptr<ptr<param>>> paramPtrVec = vector <ptr<ptr<param>>>();
	m1.modelWise([&paramPtrVec](model* m) { initParam(m, &paramPtrVec); });

	uniform_real_distribution<double> urd(-0.1, 0.1);
	default_random_engine re(44);

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

	ptr<cType> signals = make2D(7, 1);
	ptr<cType> absYGrad = make2D(7, 1);

	for (int epoch = 0; epoch < 10000000; epoch++) {

		clear2D(signals);

		for (int i = 0; i < inputs->vVector.size(); i++) {

			m1.x = inputs->vVector.at(i);
			m1.fwd();
			sub2D(m1.y, desired->vVector.at(i), m1.yGrad);
			m1.bwd();

			abs2D(m1.yGrad, absYGrad);
			add2D(signals, absYGrad, signals);

		}

		for (paramMom* p : params) {

			p->momentum = (p->beta * p->momentum) + (1 - p->beta) * p->gradient;
			p->state -= p->learnRate * p->momentum;
			p->gradient = 0;

		}

		if (epoch % 1000 == 0) {

			cout << sum1D(sum2D(abs2D(signals)))->vDouble << endl;

		}
	}

	cout << endl << "---------------------- Params:" << endl << exportParams(&paramPtrVec);

}

void trainMuMut() {

	mu m = mu(2, 7, 5);
	mu m2 = mu(5, 7, 1);

	vector<ptr<ptr<param>>> paramPtrs = vector<ptr<ptr<param>>>();
	m.modelWise([&paramPtrs](model* m) {initParam(m, &paramPtrs); });
	m2.modelWise([&paramPtrs](model* m) {initParam(m, &paramPtrs); });
	
	uniform_real_distribution<double> urd(-1, 1);
	default_random_engine re(49);


	vector<param*> params = vector<param*>();
	for (int paramIndex = 0; paramIndex < paramPtrs.size(); paramIndex++) {
		param* p = new param();
		p->learnRate = 1;
		p->state = urd(re);
		params.push_back(p);
		*paramPtrs.at(paramIndex) = p;
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

	m.prep(7);
	m.unroll(7);
	m2.prep(7);
	m2.unroll(7);

	int genSize = 40;
	genePool g(urd, re, &params, genSize, 0.003333);
	g.initParent();
	g.initChildren();
	g.populateParent();

	int bestChildIndex = 0;
	ptr<cType> bestChildCost = new cType(99999999);

	ptr<cType> childTSCost2D = make2D(7, 1);
	ptr<cType> childCost2D = make2D(7, 1);
	ptr<cType> childCost1D = make1D(1);
	ptr<cType> childCost0D = new cType();

	for (int epoch = 0; true; epoch++) {

		g.birthGeneration();
		bestChildCost->vDouble = 999999999;

		for (int childIndex = 0; childIndex < genSize; childIndex++) {
			clear2D(childCost2D);
			g.populateParams(childIndex);
			for (int tsIndex = 0; tsIndex < inputs->vVector.size(); tsIndex++) {
				m.index = 0;
				m2.index = 0;
				m.x = inputs->vVector.at(tsIndex);
				m.fwd();
				m2.x = m.y;
				m2.fwd();
				sub2D(m2.y, desired->vVector.at(tsIndex), childTSCost2D);
				abs2D(childTSCost2D, childTSCost2D);
				add2D(childCost2D, childTSCost2D, childCost2D);
			}
			sum2D(childCost2D, childCost1D);
			sum1D(childCost1D, childCost0D);

			if (childCost0D->vDouble < bestChildCost->vDouble) {
				bestChildIndex = childIndex;
				bestChildCost->vDouble = childCost0D->vDouble;
			}
		}

		g.makeParent(bestChildIndex);

		if (epoch % 10 == 0) {
			cout << bestChildCost->vDouble << endl;
		}

	}

}

void trainAttBpg() {

	attBpg a1 = attBpg(2, 1);
	muBpg m1 = muBpg(2, 10, 1);

	vector<ptr<ptr<param>>> paramPtrVec = vector <ptr<ptr<param>>>();
	a1.modelWise([&paramPtrVec](model* m) { initParam(m, &paramPtrVec); });
	m1.modelWise([&paramPtrVec](model* m) { initParam(m, &paramPtrVec); });

	uniform_real_distribution<double> urd(-0.1, 0.1);
	default_random_engine re(43);

	vector<paramMom*> params = vector<paramMom*>();
	for (ptr<ptr<param>> pptr : paramPtrVec) {
		paramMom* p = new paramMom();
		p->gradient = 0;
		p->learnRate = 0.02;
		p->state = urd(re);
		p->beta = 0.0;
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

	a1.prep(7, 7);
	m1.prep(7);

	a1.unroll(7, 7);
	m1.unroll(7);
	m1.x = make2D(7, 2);

	m1.yGrad = make2D(7, 1);

	for (int epoch = 0; epoch < 20000; epoch++) {

		for (int i = 0; i < inputs->vVector.size(); i++) {

			ptr<cType> tsInputs = inputs->vVector.at(i);
			a1.x = tsInputs;
			for (int j = 0; j < m1.size(); j++) {

				attTSBpg* a = (attTSBpg*)a1.at(j).get();
				a1.hTIn->vVector.at(j) = m1.hTOut;
				a1.incFwd(1);
				m1.x->vVector[m1.index] = a->y;
				m1.incFwd(1);

			}
			sub2D(m1.y, desired->vVector.at(i), m1.yGrad);
			for (int j = m1.size() - 1; j >= 0; j--) {

				attTSBpg* a = (attTSBpg*)a1.at(j).get();
				m1.incBwd(1);
				a1.yGrad->vVector.at(j) = m1.xGrad->vVector.at(j);
				a1.incBwd(1);
				add1D(m1.hTInGrad, a1.hTInGrad->vVector.at(j), m1.hTInGrad);

			}

		}

		for (paramMom* p : params) {
			p->momentum = p->beta * p->momentum + (1 - p->beta) * p->gradient;
			p->state -= p->learnRate * p->momentum;
			p->gradient = 0;
		}

		if (epoch % 1000 == 0) {

			cout << sum1D(sum2D(abs2D(m1.yGrad)))->vDouble << endl;

		}

	}

	cout << endl << "---------------------- Params:" << endl << exportParams(&paramPtrVec);


}

void trainAccelerator() {

	ptr<model> nlr = neuronLRBpg(0.3);

	// initialize the training subject models
	lstmBpg subLstm(7);
	syncBpg subSyncIn(tnnBpg({ 2, 7 }, nlr));
	syncBpg subSyncOut(tnnBpg({ 7, 1 }, nlr));

	// initialize the accelerator models
	muTS* accMuTS = new muTS(1, 10, 1, tnn({1 + 1, 10 + 1}, nlr));
	sync accSync(accMuTS);

	vector<ptr<ptr<param>>> subParamPtrVec = vector<ptr<ptr<param>>>();
	subLstm.modelWise([&subParamPtrVec](model* m) { initParam(m, &subParamPtrVec); });
	subSyncIn.modelWise([&subParamPtrVec](model* m) { initParam(m, &subParamPtrVec); });
	subSyncOut.modelWise([&subParamPtrVec](model* m) { initParam(m, &subParamPtrVec); });

	vector<ptr<ptr<param>>> accParamPtrVec = vector<ptr<ptr<param>>>();
	accSync.modelWise([&accParamPtrVec](model* m) { initParam(m, &accParamPtrVec); });

	uniform_real_distribution<double> urd(-1, 1);
	default_random_engine re(46);

	vector<paramMom*> subParamVec = vector<paramMom*>();
	for (ptr<ptr<param>> pptr : subParamPtrVec) {

		paramMom* param = new paramMom();
		param->beta = 0.9;
		param->learnRate = 0.02;
		param->state = urd(re);
		*pptr = param;
		subParamVec.push_back(param);

	}

	vector<param*> accParamVec = vector<param*>();
	for (ptr<ptr<param>> pptr : accParamPtrVec) {

		param* p = new param();
		p->learnRate = 0.02;
		p->state = urd(re);
		*pptr = p;
		accParamVec.push_back(p);

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

	subLstm.prep(4);
	subLstm.unroll(4);
	subSyncIn.prep(4);
	subSyncIn.unroll(4);
	subSyncOut.prep(4);
	subSyncOut.unroll(4);
	
	function<paramMom(paramMom*)> saveParam = [](paramMom* p) { return *p; };
	function<void(paramMom&, paramMom*&)> rollBackParam = [](paramMom& saved, paramMom*& changed) { *changed = saved; };

	// Save values of the parameters at epoch == 0
	vector<paramMom> InitialValues = getVals(&subParamVec, saveParam);

	accSync.prep(subParamVec.size());
	accSync.unroll(subParamVec.size());

	int accGenSize = 40;
	double accMutProb = 0.2;
	genePool accGp = genePool(urd, re, &accParamVec, accGenSize, accMutProb);
	accGp.initParent();
	accGp.initChildren();
	accGp.populateParent();
	
	subSyncOut.yGrad = make2D(4, 1);
	ptr<cType> subTSCost2D = make2D(4, 1);
	ptr<cType> subCost2D = make2D(4, 1);
	ptr<cType> subEpochCost2D = make2D(4, 1);

	double childCost;
	double bestChildCost;
	int bestChildIndex;

	// train accelerator
	for (int epoch = 0; epoch < 100; epoch++) {

		childCost = 0;
		bestChildCost = 9999999;
		bestChildIndex = 0;


		accGp.birthGeneration();

		for (int childIndex = 0; childIndex < accGenSize; childIndex++) {

			accGp.populateParams(childIndex);

			// Roll back the values of the parameters to equal those at epoch == 0
			setVals(&InitialValues, &subParamVec, rollBackParam);

			// clear all subMuTSs' cTIns and hTIns
			elemWise<ptr<model>>(&accSync, [](ptr<model> p) {
				muTS* m = (muTS*)p.get();
				clear1D(m->hTIn);
				clear1D(m->cTIn);
			});

			// clear the accumulated cost for the subModel over all epochs
			clear2D(subCost2D);

			// train subject
			for (int subEpoch = 0; subEpoch < 10000; subEpoch++) {

				// clear the cost for the epoch
				clear2D(subEpochCost2D);

				for (int tsIndex = 0; tsIndex < inputs->vVector.size(); tsIndex++) {

					// carry forward state
					subSyncIn.x = inputs->vVector.at(tsIndex);
					subSyncIn.fwd();
					subLstm.x = subSyncIn.y;
					subLstm.fwd();
					subSyncOut.x = subLstm.y;
					subSyncOut.fwd();

					// calculate output gradient of subSyncOut
					sub2D(subSyncOut.y, desired->vVector.at(tsIndex), subSyncOut.yGrad);

					// carry backward gradient
					subSyncOut.bwd();
					subLstm.yGrad = subSyncOut.xGrad;
					subLstm.bwd();
					subSyncIn.yGrad = subLstm.xGrad;
					subSyncIn.bwd();

					// add output gradient to total epoch cost
					abs2D(subSyncOut.yGrad, subTSCost2D);
					add2D(subEpochCost2D, subTSCost2D, subEpochCost2D);

				}

				add2D(subCost2D, subEpochCost2D, subCost2D);

				for (paramMom* p : subParamVec) {
					p->momentum = p->beta * p->momentum + (1 - p->beta) * p->gradient;
					p->state -= p->learnRate * p->momentum;
					p->gradient = 0;
				}

				if (subEpoch % 100 == 0) {
					double childEpochCost = sum1D(sum2D(subEpochCost2D))->vDouble;
					childCost = sum1D(sum2D(subCost2D))->vDouble;
					cout << childEpochCost << " / " << childCost << endl;
					if (isnan(childCost)) {
						childCost = 9999999;
					}

					// modify the learn rates of all subParameters
					for (int muIndex = 0; muIndex < accSync.size(); muIndex++) {

						int learnRateModifyIndex = subEpoch / 100;

						muTS* m = (muTS*)accSync.at(muIndex).get();
						m->x = new cType{ subParamVec.at(muIndex)->learnRate/*, (double)muIndex*/ };
						m->fwd();
						copy1D(m->cTOut, m->cTIn);
						copy1D(m->hTOut, m->hTIn);
						subParamVec.at(muIndex)->learnRate += 0.002 * tanh(m->y->vVector.at(0)->vDouble);
					}

				}

			}

			if (childCost < bestChildCost) {
				bestChildCost = childCost;
				bestChildIndex = childIndex;
			}

			cout << "----------------------" << endl << "END SUBJECT TRAINING" << endl << "ACCELERATOR CHILD EPOCH: " << childIndex << endl << "----------------------" << endl;

		}

		accGp.makeParent(bestChildIndex);

	}

	



}

void trainDigitRecognizer() {

	vector<string> paths = { 
	"D:\\files\\programming\\GitHub\\aurora\\dev\\windows\\AuroraCPP\\AuroraCPP\\Debug\\0004_CH4M\\0", 
	"D:\\files\\programming\\GitHub\\aurora\\dev\\windows\\AuroraCPP\\AuroraCPP\\Debug\\0004_CH4M\\1",
	"D:\\files\\programming\\GitHub\\aurora\\dev\\windows\\AuroraCPP\\AuroraCPP\\Debug\\0004_CH4M\\2",
	"D:\\files\\programming\\GitHub\\aurora\\dev\\windows\\AuroraCPP\\AuroraCPP\\Debug\\0004_CH4M\\3",
	"D:\\files\\programming\\GitHub\\aurora\\dev\\windows\\AuroraCPP\\AuroraCPP\\Debug\\0004_CH4M\\4",
	"D:\\files\\programming\\GitHub\\aurora\\dev\\windows\\AuroraCPP\\AuroraCPP\\Debug\\0004_CH4M\\5",
	"D:\\files\\programming\\GitHub\\aurora\\dev\\windows\\AuroraCPP\\AuroraCPP\\Debug\\0004_CH4M\\6",
	"D:\\files\\programming\\GitHub\\aurora\\dev\\windows\\AuroraCPP\\AuroraCPP\\Debug\\0004_CH4M\\7",
	"D:\\files\\programming\\GitHub\\aurora\\dev\\windows\\AuroraCPP\\AuroraCPP\\Debug\\0004_CH4M\\8",
	"D:\\files\\programming\\GitHub\\aurora\\dev\\windows\\AuroraCPP\\AuroraCPP\\Debug\\0004_CH4M\\9" };

	ptr<cType> inputs = new cType();
	for (int i = 0; i < paths.size(); i++) {

		directory_iterator d(paths.at(i));
		ptr<cType> trainingSets = new cType();
		
		ifstream ifs;

		for (const auto& entry : d) {
			
			path p = entry.path();
			ifstream file(p);
			char data[10000];
			file.read(data, 10000);
		}
	}


	ofstream myfile;
	myfile.open("example.txt");
	myfile << "Writing this to a file.\n";
	myfile.close();

}

string exportParams(vector<ptr<ptr<param>>>* paramPtrVec) {
	string result = "";
	for (int i = 0; i < paramPtrVec->size(); i++) {
		result += to_string((*paramPtrVec->at(i))->state);
		result += "\n";
	}
	return result;
}