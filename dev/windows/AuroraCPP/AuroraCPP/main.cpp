#pragma once
#include "main.h"
#include "general.h"
#include <fstream>
#include <filesystem>
#include <FreeImage.h>

using namespace std;
using namespace filesystem;

string exportParams(vector<ptr<ptr<param>>>* paramPtrVec);
void trainTnnBpg();
void trainTnnMut();
void trainSyncBpg();
void trainLstmBpg();
void trainMuBpg();
void trainAttBpg();
void trainDigitRecognizer();

ptr<cType> somewhatNormalDistribution(double verStretch, double horStretch, double upShift, double min, double max, double inc)
{
	ptr<cType> result = new cType();
	for (double x = min; x <= max; x += inc) {
		double dvals = verStretch * (horStretch * (1 - pow(tanh(x), 2))) + upShift;
		int vals = (int)dvals;
		for (int i = 0; i < vals; i++) {
			result->vVector.push_back(new cType(x));
		}
	}
	return result;
}

int main() {
	trainTnnMut();
	return 0;
}

void trainTnnBpg() {

	ptr<model> nlr = neuronLRBpg(0.05);
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

void trainTnnMut() {

	ptr<model> nlr = neuronLR(0.05);
	ptr<seq> s = tnn({ 2, 5, 1 }, { nlr, nlr, nlr });

	vector <ptr<ptr<param>>> paramPtrVec = vector <ptr<ptr<param>>>();
	s->modelWise([&paramPtrVec](model* m) { initParam(m, &paramPtrVec); });

	// set up random engine for initializing param states
	uniform_real_distribution<double> u(-1, 1);
	default_random_engine re(26);

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

		if (epoch % 10 == 0) {
			double bcc = bestChildCost->vVector.at(0)->vDouble;
			cout << bcc << endl;
		}

	}


	return;

}

void trainSyncBpg() {

	ptr<model> nlr = neuronLRBpg(0.05);
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

void trainLstmBpg() {

	ptr<model> nlr = neuronLRBpg(0.05);

	lstmBpg l1 = lstmBpg(5);
	ptr<model> inNN = tnnBpg({ 2, 5 }, nlr);
	ptr<model> outNN = tnnBpg({ 5, 1 }, nlr);

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

void trainAttBpg() {

	attBpg a1 = attBpg(2, 1);
	muBpg m1 = muBpg(2, 12, 1);

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