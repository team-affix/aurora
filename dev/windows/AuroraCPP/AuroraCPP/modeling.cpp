#pragma once
#include "modeling.h"

#pragma region functions

#pragma region external
seq* tnn(vector<int> npl, vector<model*> layerNeuronTemplates) {
	seq* result = new seq();
	for (int i = 0; i < npl.size() - 1; i++) {
		result->push_back(new layer(npl.at(i), layerNeuronTemplates.at(i)));
		result->push_back(new wJunc(npl.at(i), npl.at(i + 1)));
	}
	result->push_back(new layer(npl.back(), layerNeuronTemplates.back()));
	return result;
}
seqBpg* tnnBpg(vector<int> npl, vector<model*> layerNeuronTemplates) {
	seqBpg* result = new seqBpg();
	for (int i = 0; i < npl.size() - 1; i++) {
		result->push_back(new layerBpg(npl.at(i), layerNeuronTemplates.at(i)));
		result->push_back(new wJuncBpg(npl.at(i), npl.at(i + 1)));
	}
	result->push_back(new layerBpg(npl.back(), layerNeuronTemplates.back()));
	return result;
}
seq* tnn(vector<int> npl, model* neuronTemplate) {
	seq* result = new seq();
	for (int i = 0; i < npl.size() - 1; i++) {
		result->push_back(new layer(npl.at(i), neuronTemplate));
		result->push_back(new wJunc(npl.at(i), npl.at(i + 1)));
	}
	result->push_back(new layer(npl.back(), neuronTemplate));
	return result;
}
seqBpg* tnnBpg(vector<int> npl, model* neuronTemplate) {
	seqBpg* result = new seqBpg();
	for (int i = 0; i < npl.size() - 1; i++) {
		result->push_back(new layerBpg(npl.at(i), neuronTemplate));
		result->push_back(new wJuncBpg(npl.at(i), npl.at(i + 1)));
	}
	result->push_back(new layerBpg(npl.back(), neuronTemplate));
	return result;
}
seq neuronSm() {

	// construct tanh neuron
	seq nsm = seq();
	nsm.push_back(new bias());
	nsm.push_back(new act(new actFuncSm()));
	return nsm;

}
seq neuronTh() {

	// construct tanh neuron
	seq nth = seq();
	nth.push_back(new bias());
	nth.push_back(new act(new actFuncTh()));
	return nth;

}
seq neuronLR(double m) {

	// construct tanh neuron
	seq nlr = seq();
	nlr.push_back(new bias());
	nlr.push_back(new act(new actFuncLR(m)));
	return nlr;

}
seqBpg neuronSmBpg() {

	// construct tanh neuron
	seqBpg nsm = seqBpg();
	nsm.push_back(new biasBpg());
	nsm.push_back(new actBpg(new actFuncSm()));
	return nsm;

}
seqBpg neuronThBpg() {

	// construct tanh neuron
	seqBpg nth = seqBpg();
	nth.push_back(new biasBpg());
	nth.push_back(new actBpg(new actFuncTh()));
	return nth;

}
seqBpg neuronLRBpg(double m) {

	// construct tanh neuron
	seqBpg nlr = seqBpg();
	nlr.push_back(new biasBpg());
	nlr.push_back(new actBpg(new actFuncLR(m)));
	return nlr;

}
void initParam(model* m, vector<sPtr<sPtr<param>>>* paramVecOutput) {
	if (bias* b = dynamic_cast<bias*>(m)) {
		b->prm = new sPtr<param>();
		paramVecOutput->push_back(b->prm);
	}
	else if (biasBpg* b = dynamic_cast<biasBpg*>(m)) {
		b->prm = new sPtr<param>();
		paramVecOutput->push_back(b->prm);
	}
	else if (weight* w = dynamic_cast<weight*>(m)) {
		w->prm = new sPtr<param>();
		paramVecOutput->push_back(w->prm);
	}
	else if (weightBpg* w = dynamic_cast<weightBpg*>(m)) {
		w->prm = new sPtr<param>();
		paramVecOutput->push_back(w->prm);
	}
}
#pragma endregion
#pragma region model
void modelFwd(sPtr<cType> x, sPtr<cType> y) {
	*y = *x;
}
void modelBwd(sPtr<cType> yGrad, sPtr<cType> xGrad) {
	*xGrad = *yGrad;
}
#pragma endregion
#pragma region bias
void biasFwd(sPtr<cType> x, sPtr<cType> y, sPtr<param> prm) {
	y->vDouble = x->vDouble + prm->state;
}
void biasBwd(sPtr<cType> yGrad, sPtr<cType> xGrad, sPtr<param> prm) {
	paramSgd* p = (paramSgd*)prm.get();
	p->gradient += yGrad->vDouble;
	xGrad->vDouble = yGrad->vDouble;
}
#pragma endregion
#pragma region act
void actFwd(sPtr<cType> x, sPtr<cType> y, sPtr<actFunc> af) {
	y->vDouble = af->eval(x->vDouble);
}
void actBwd(sPtr<cType> yGrad, sPtr<cType> y, sPtr<cType> xGrad, sPtr<actFunc> af) {
	xGrad->vDouble = yGrad->vDouble * af->deriv(y->vDouble);
}
#pragma endregion
#pragma region weight
void weightFwd(sPtr<cType> x, sPtr<cType> y, sPtr<param> prm) {
	y->vDouble = x->vDouble * prm->state;
}
void weightBwd(sPtr<cType> yGrad, sPtr<cType> xGrad, sPtr<cType> x, sPtr<param> prm) {
	paramSgd* p = (paramSgd*)prm.get();
	p->gradient += yGrad->vDouble * x->vDouble;
	xGrad->vDouble = yGrad->vDouble * prm->state;
}
#pragma endregion
#pragma region wSet
void wSetFwd(sPtr<cType> x, sPtr<cType> y, vector<sPtr<model>>* models) {

	vector<sPtr<cType>>* yVec = &y->vVector;
	for (int i = 0; i < models->size(); i++) {

		model* m = models->at(i).get();
		m->x->vDouble = x->vDouble;
		m->fwd();
		yVec->at(i)->vDouble = m->y->vDouble;

	}

}
void wSetBwd(sPtr<cType> yGrad, sPtr<cType> xGrad, vector<sPtr<model>>* models) {

	// reset xGrad of weightSet to avoid overaccumulation
	xGrad->vDouble = 0;

	vector<sPtr<cType>>* yGradVec = &yGrad->vVector;
	for (int i = 0; i < models->size(); i++) {

		modelBpg* m = (modelBpg*)models->at(i).get();
		m->yGrad->vDouble = yGradVec->at(i)->vDouble;
		m->bwd();
		xGrad->vDouble += m->xGrad->vDouble;

	}

}
void wSetModelWise(function<void(model*)> func, vector<sPtr<model>>* models) {
	for (int i = 0; i < models->size(); i++) {
		models->at(i)->modelWise(func);
	}
}
#pragma endregion
#pragma region wJunc
void wJuncFwd(sPtr<cType> x, sPtr<cType> y, vector<sPtr<model>>* models) {

	// reset all values in vector to zero, to ensure that no overaccumulation occurs
	clear1D(y);
	vector<sPtr<cType>>* xVec = &x->vVector;
	for (int i = 0; i < models->size(); i++) {

		model* m = models->at(i).get();
		m->x->vDouble = xVec->at(i)->vDouble;
		m->fwd();

		// accumulates output
		add1D(y, m->y, y);

	}

}
void wJuncBwd(sPtr<cType> yGrad, sPtr<cType> xGrad, vector<sPtr<model>>* models) {

	vector<sPtr<cType>>* xGradVec = &xGrad->vVector;
	for (int i = 0; i < models->size(); i++) {

		modelBpg* m = (modelBpg*)models->at(i).get();
		m->yGrad->vVector = yGrad->vVector;
		m->bwd();
		xGradVec->at(i)->vDouble = m->xGrad->vDouble;

	}

}
void wJuncModelWise(function<void(model*)> func, vector<sPtr<model>>* models) {
	for (int i = 0; i < models->size(); i++) {
		models->at(i)->modelWise(func);
	}
}
#pragma endregion
#pragma region seq
void seqFwd(sPtr<cType> x, sPtr<cType> y, vector<sPtr<model>>* models) {

	sPtr<cType> currentInput = x;
	for (int i = 0; i < models->size(); i++) {

		model* m = models->at(i).get();
		m->x = currentInput;
		m->fwd();
		currentInput = m->y;

	}
	*y = *currentInput;

}
void seqBwd(sPtr<cType> yGrad, sPtr<cType> xGrad, vector<sPtr<model>>* models) {

	sPtr<cType> currentGradient = yGrad;
	for (int i = models->size() - 1; i >= 0; i--) {

		modelBpg* m = (modelBpg*)models->at(i).get();
		m->yGrad = currentGradient;
		m->bwd();
		currentGradient = m->xGrad;

	}
	*xGrad = *currentGradient;

}
void seqModelWise(function<void(model*)> func, vector<sPtr<model>>* models) {

	for (int i = 0; i < models->size(); i++) {

		models->at(i)->modelWise(func);

	}

}
#pragma endregion
#pragma region layer
void layerFwd(sPtr<cType> x, sPtr<cType> y, vector<sPtr<model>>* models) {

	vector<sPtr<cType>>* xVec = &x->vVector;
	vector<sPtr<cType>>* yVec = &y->vVector;
	for (int i = 0; i < models->size(); i++) {

		model* m = models->at(i).get();
		m->x = xVec->at(i);
		m->fwd();
		yVec->at(i) = m->y;

	}

}
void layerBwd(sPtr<cType> yGrad, sPtr<cType> xGrad, vector<sPtr<model>>* models) {

	vector<sPtr<cType>>* xGradVec = &xGrad->vVector;
	vector<sPtr<cType>>* yGradVec = &yGrad->vVector;
	for (int i = 0; i < models->size(); i++) {

		modelBpg* m = (modelBpg*)models->at(i).get();
		m->yGrad = yGradVec->at(i);
		m->bwd();
		xGradVec->at(i) = m->xGrad;

	}

}
void layerModelWise(function<void(model*)> func, vector<sPtr<model>>* models) {
	for (int i = 0; i < models->size(); i++) {
		models->at(i)->modelWise(func);
	}
}
#pragma endregion
#pragma region sync
void syncFwd(sPtr<cType> x, sPtr<cType> y, vector<sPtr<model>>* models) {
	layerFwd(x, y, models);
}
void syncBwd(sPtr<cType> yGrad, sPtr<cType> xGrad, vector<sPtr<model>>* models) {
	layerBwd(yGrad, xGrad, models);
}
void syncModelWise(function<void(model*)> func, vector<sPtr<model>>* models) {
	for (int i = 0; i < models->size(); i++) {
		models->at(i)->modelWise(func);
	}
}
#pragma endregion
#pragma region lstmTS
void lstmTSFwd(sPtr<cType> x, sPtr<cType> cTIn, sPtr<cType> hTIn,
	sPtr<cType> y, sPtr<cType> cTOut, sPtr<cType> hTOut, 
	sPtr<model> aGate, sPtr<model> bGate, sPtr<model> cGate, sPtr<model> dGate) {
	
	sPtr<cType> input = concat(x, hTIn);

	aGate->x = input;
	bGate->x = input;
	cGate->x = input;
	dGate->x = input;

	aGate->fwd();
	bGate->fwd();
	cGate->fwd();
	dGate->fwd();

	sPtr<cType> cT = mult1D(aGate->y, cTIn);
	add1D(cT, mult1D(bGate->y, cGate->y), cT);
	cTOut->vVector = cT->vVector;
	sPtr<cType> hT = mult1D(cT, dGate->y);
	hTOut->vVector = hT->vVector;
	y->vVector = hT->vVector;

}
void lstmTSBwd(int units,
	sPtr<cType> cTOut, sPtr<cType> cTIn,
	sPtr<cType> yGrad, sPtr<cType> cTOutGrad, sPtr<cType> hTOutGrad, 
	sPtr<cType> xGrad, sPtr<cType> cTInGrad, sPtr<cType> hTInGrad,
	sPtr<model> aGate, sPtr<model> bGate, sPtr<model> cGate, sPtr<model> dGate) {

	// cast all gates to the modelBpgs that they are
	modelBpg* aGateBpg = (modelBpg*)aGate.get();
	modelBpg* bGateBpg = (modelBpg*)bGate.get();
	modelBpg* cGateBpg = (modelBpg*)cGate.get();
	modelBpg* dGateBpg = (modelBpg*)dGate.get();

	// calculate major gradient tracks
	sPtr<cType> hTGrad = add1D(hTOutGrad, yGrad);
	sPtr<cType> cTGrad = add1D(cTOutGrad, mult1D(dGateBpg->y, hTGrad));

	// calculate each gate's output gradient
	dGateBpg->yGrad = mult1D(hTGrad, cTOut);
	cGateBpg->yGrad = mult1D(bGateBpg->y, cTGrad);
	bGateBpg->yGrad = mult1D(cGateBpg->y, cTGrad);
	aGateBpg->yGrad = mult1D(cTGrad, cTIn);

	mult1D(aGateBpg->y, cTGrad, cTGrad);

	// carry each gates' gradient backward
	aGateBpg->bwd();
	bGateBpg->bwd();
	cGateBpg->bwd();
	dGateBpg->bwd();

	sPtr<cType> inputGrad = add1D(add1D(aGateBpg->xGrad, bGateBpg->xGrad), add1D(cGateBpg->xGrad, dGateBpg->xGrad));
	vector<sPtr<cType>> xGradVec(inputGrad->vVector.begin(), inputGrad->vVector.begin() + units);
	vector<sPtr<cType>> hTInGradVec(inputGrad->vVector.begin() + units, inputGrad->vVector.begin() + units * 2);
	
	xGrad->vVector = xGradVec;
	hTInGrad->vVector = hTInGradVec;
	cTInGrad->vVector = cTGrad->vVector;

}
void lstmTSModelWise(function<void(model*)> func, sPtr<model> aGate, sPtr<model> bGate, sPtr<model> cGate, sPtr<model> dGate) {
	aGate->modelWise(func);
	bGate->modelWise(func);
	cGate->modelWise(func);
	dGate->modelWise(func);
}
#pragma endregion
#pragma region lstm
void lstmModelWise(function<void(model*)> func, sPtr<model> lstmTSTemplate) {

	lstmTSTemplate->modelWise(func);

}
#pragma endregion

#pragma endregion

#pragma region definitions

#pragma region model
model::model() {
	x = new cType(0);
	y = new cType(0);
}
void model::fwd() {
	modelFwd(x, y);
}
void model::modelWise(function<void(model*)> func) {
	func(this);
}
sPtr<model> model::clone() {
	return new model();
}
modelBpg::modelBpg() {
	x = new cType(0);
	y = new cType(0);
	xGrad = new cType(0);
	yGrad = new cType(0);
}
void modelBpg::bwd() {
	modelBwd(yGrad, xGrad);
}
sPtr<model> modelBpg::clone() {
	return new modelBpg(*this);
}
#pragma endregion
#pragma region bias
bias::bias() {
	x = new cType(0);
	y = new cType(0);
}
void bias::fwd() {
	biasFwd(x, y, *prm);
}
sPtr<model> bias::clone() {
	bias* result = new bias();
	result->x = new cType(*x);
	result->y = new cType(*y);
	result->prm = prm;
	return result;
}
biasBpg::biasBpg() {
	x = new cType(0);
	y = new cType(0);
	xGrad = new cType(0);
	yGrad = new cType(0);
}
void biasBpg::fwd() {
	biasFwd(x, y, *prm);
}
void biasBpg::bwd() {
	biasBwd(yGrad, xGrad, *prm);
}
sPtr<model> biasBpg::clone() {
	biasBpg* result = new biasBpg();
	result->x = new cType(*x);
	result->y = new cType(*y);
	result->xGrad = new cType(*xGrad);
	result->yGrad = new cType(*yGrad);
	result->prm = prm;
	return result;
}
#pragma endregion
#pragma region act
act::act() {

}
act::act(actFunc* _af) {
	x = new cType(0);
	y = new cType(0);
	this->af = _af;
}
act::act(sPtr<actFunc> _af) {
	x = new cType(0);
	y = new cType(0);
	this->af = _af;
}
void act::fwd() {
	actFwd(x, y, af);
}
sPtr<model> act::clone() {
	act* result = new act();
	result->x = new cType(*x);
	result->y = new cType(*y);
	result->af = af;
	return result;
}
actBpg::actBpg() {

}
actBpg::actBpg(actFunc* _af) {
	x = new cType(0);
	y = new cType(0);
	xGrad = new cType(0);
	yGrad = new cType(0);
	this->af = _af;
}
actBpg::actBpg(sPtr<actFunc> _af) {
	x = new cType(0);
	y = new cType(0);
	this->af = _af;
}
void actBpg::fwd() {
	actFwd(x, y, af);
}
void actBpg::bwd() {
	actBwd(yGrad, y, xGrad, af);
}
sPtr<model> actBpg::clone() {
	actBpg* result = new actBpg();
	result->x = new cType(*x);
	result->y = new cType(*y);
	result->xGrad = new cType(*xGrad);
	result->yGrad = new cType(*yGrad);
	result->af = af;
	return result;
}
#pragma endregion
#pragma region weight
weight::weight() {
	x = new cType(0);
	y = new cType(0);
}
void weight::fwd() {
	weightFwd(x, y, *prm);
}
sPtr<model> weight::clone() {
	weight* result = new weight();
	result->x = new cType(*x);
	result->y = new cType(*y);
	result->prm = prm;
	return result;
}
weightBpg::weightBpg() {
	x = new cType(0);
	y = new cType(0);
	xGrad = new cType(0);
	yGrad = new cType(0);
}
void weightBpg::fwd() {
	weightFwd(x, y, *prm);
}
void weightBpg::bwd() {
	weightBwd(yGrad, xGrad, x, *prm);
}
sPtr<model> weightBpg::clone() {
	weightBpg* result = new weightBpg();
	result->x = new cType(*x);
	result->y = new cType(*y);
	result->xGrad = new cType(*xGrad);
	result->yGrad = new cType(*yGrad);
	result->prm = prm;
	return result;
}
#pragma endregion
#pragma region wSet
wSet::wSet() {

}
wSet::wSet(int _a) {
	this->a = _a;
	x = new cType(0);
	y = new cType({});
	for (int i = 0; i < _a; i++) {
		y->vVector.push_back(new cType(0));
		push_back(new weight());
	}
}
void wSet::fwd() {
	wSetFwd(x, y, this);
}
void wSet::modelWise(function<void(model*)> func) {
	func(this);
	wSetModelWise(func, this);
}
sPtr<model> wSet::clone() {
	wSet* result = new wSet();
	result->x = new cType(*x);
	result->y = new cType(*y);
	for (int i = 0; i < a; i++) {
		result->push_back(at(i)->clone());
	}
	return result;
}
wSetBpg::wSetBpg() {

}
wSetBpg::wSetBpg(int _a) {
	this->a = _a;
	x = new cType(0);
	y = new cType({});
	xGrad = new cType(0);
	yGrad = new cType({});
	for (int i = 0; i < _a; i++) {
		y->vVector.push_back(new cType(0));
		yGrad->vVector.push_back(new cType(0));
		push_back(new weightBpg());
	}
}
void wSetBpg::fwd() {
	wSetFwd(x, y, this);
}
void wSetBpg::bwd() {
	wSetBwd(yGrad, xGrad, this);
}
void wSetBpg::modelWise(function<void(model*)> func) {
	func(this);
	wSetModelWise(func, this);
}
sPtr<model> wSetBpg::clone() {
	wSetBpg* result = new wSetBpg();
	result->x = new cType(*x);
	result->y = new cType(*y);
	result->xGrad = new cType(*xGrad);
	result->yGrad = new cType(*yGrad);
	for (int i = 0; i < a; i++) {
		result->push_back(at(i)->clone());
	}
	return result;
}
#pragma endregion
#pragma region wJunc
wJunc::wJunc() {

}
wJunc::wJunc(int _a, int _b) {
	this->a = _a;
	this->b = _b;

	x = new cType({});
	y = new cType({});

	for (int i = 0; i < _a; i++) {
		x->vVector.push_back(new cType(0));
		push_back(new wSet(b));
	}

	for (int i = 0; i < _b; i++) {
		y->vVector.push_back(new cType(0));
	}

}
void wJunc::fwd() {
	wJuncFwd(x, y, this);
}
void wJunc::modelWise(function<void(model*)> func) {
	func(this);
	wJuncModelWise(func, this);
}
sPtr<model> wJunc::clone() {
	wJunc* result = new wJunc();
	result->x = new cType(*x);
	result->y = new cType(*y);
	for (int i = 0; i < a; i++) {
		result->push_back(at(i)->clone());
	}
	return result;
}
wJuncBpg::wJuncBpg() {

}
wJuncBpg::wJuncBpg(int _a, int _b) {

	this->a = _a;
	this->b = _b;

	x = new cType({});
	y = new cType({});
	xGrad = new cType({});
	yGrad = new cType({});

	for (int i = 0; i < _a; i++) {
		x->vVector.push_back(new cType(0));
		xGrad->vVector.push_back(new cType(0));
		push_back(new wSetBpg(b));
	}

	for (int i = 0; i < _b; i++) {
		y->vVector.push_back(new cType(0));
		yGrad->vVector.push_back(new cType(0));
	}

}
void wJuncBpg::fwd() {
	wJuncFwd(x, y, this);
}
void wJuncBpg::bwd() {
	wJuncBwd(yGrad, xGrad, this);
}
void wJuncBpg::modelWise(function<void(model*)> func) {
	func(this);
	wJuncModelWise(func, this);
}
sPtr<model> wJuncBpg::clone() {
	wJuncBpg* result = new wJuncBpg();
	result->x = new cType(*x);
	result->y = new cType(*y);
	result->xGrad = new cType(*xGrad);
	result->yGrad = new cType(*yGrad);
	for (int i = 0; i < a; i++) {
		result->push_back(at(i)->clone());
	}
	return result;
}
#pragma endregion
#pragma region seq
seq::seq() {
	x = new cType(0);
	y = new cType(0);
}
void seq::fwd() {
	seqFwd(x, y, this);
}
void seq::modelWise(function<void(model*)> func) {
	func(this);
	seqModelWise(func, this);
}
sPtr<model> seq::clone() {
	seq* result = new seq();
	result->x = new cType(*x);
	result->y = new cType(*y);
	for (int i = 0; i < size(); i++) {
		result->push_back(at(i)->clone());
	}
	return result;
}
seqBpg::seqBpg() {
	x = new cType(0);
	y = new cType(0);
	xGrad = new cType(0);
	yGrad = new cType(0);
}
void seqBpg::fwd() {
	seqFwd(x, y, this);
}
void seqBpg::bwd() {
	seqBwd(yGrad, xGrad, this);
}
void seqBpg::modelWise(function<void(model*)> func) {
	func(this);
	seqModelWise(func, this);
}
sPtr<model> seqBpg::clone() {
	seqBpg* result = new seqBpg();
	result->x = new cType(*x);
	result->y = new cType(*y);
	result->xGrad = new cType(*xGrad);
	result->yGrad = new cType(*yGrad);
	for (int i = 0; i < size(); i++) {
		result->push_back(at(i)->clone());
	}
	return result;
}
#pragma endregion
#pragma region layer
layer::layer() {

	x = new cType({});
	y = new cType({});

}
layer::layer(int a, model* modelTemplate) {

	x = new cType({});
	y = new cType({});

	for (int i = 0; i < a; i++) {

		// initialize the x, y cTypes as vectors to insure that all members of the cType are initialized
		x->vVector.push_back(new cType({}));
		y->vVector.push_back(new cType({}));

		push_back(modelTemplate->clone());

	}

}
layer::layer(int a, sPtr<model> modelTemplate) {

	x = new cType({});
	y = new cType({});

	for (int i = 0; i < a; i++) {

		// initialize the x, y cTypes as vectors to insure that all members of the cType are initialized
		x->vVector.push_back(new cType({}));
		y->vVector.push_back(new cType({}));
		
		push_back(modelTemplate->clone());

	}

}
void layer::fwd() {
	layerFwd(x, y, this);
}
void layer::modelWise(function<void(model*)> func) {
	func(this);
	layerModelWise(func, this);
}
sPtr<model> layer::clone() {
	layer* result = new layer();
	result->x = new cType(*x);
	result->y = new cType(*y);
	for (int i = 0; i < size(); i++) {
		result->push_back(at(i)->clone());
	}
	return result;
}
layerBpg::layerBpg() {

	// initialized the x, y, xGrad, yGrad cTypes as vectors to insure that all members of the cType are initialized
	x = new cType({});
	y = new cType({});
	xGrad = new cType({});
	yGrad = new cType({});

}
layerBpg::layerBpg(int a, model* modelTemplate) {

	x = new cType({});
	y = new cType({});
	xGrad = new cType({});
	yGrad = new cType({});

	for (int i = 0; i < a; i++) {

		// initialize the x, y, xGrad, yGrad cTypes as vectors to insure that all members of the cType are initialized
		x->vVector.push_back(new cType({}));
		y->vVector.push_back(new cType({}));
		xGrad->vVector.push_back(new cType({}));
		yGrad->vVector.push_back(new cType({}));

		push_back(modelTemplate->clone());

	}

}
layerBpg::layerBpg(int a, sPtr<model> modelTemplate) {

	x = new cType({});
	y = new cType({});
	xGrad = new cType({});
	yGrad = new cType({});

	for (int i = 0; i < a; i++) {

		// initialize the x, y, xGrad, yGrad cTypes as vectors to insure that all members of the cType are initialized
		x->vVector.push_back(new cType({}));
		y->vVector.push_back(new cType({}));
		xGrad->vVector.push_back(new cType({}));
		yGrad->vVector.push_back(new cType({}));

		push_back(modelTemplate->clone());

	}

}
void layerBpg::fwd() {
	layerFwd(x, y, this);
}
void layerBpg::bwd() {
	layerBwd(yGrad, xGrad, this);
}
void layerBpg::modelWise(function<void(model*)> func) {
	func(this);
	layerModelWise(func, this);
}
sPtr<model> layerBpg::clone() {
	layerBpg* result = new layerBpg();
	result->x = new cType(*x);
	result->y = new cType(*y);
	result->xGrad = new cType(*xGrad);
	result->yGrad = new cType(*yGrad);
	for (int i = 0; i < size(); i++) {
		result->push_back(at(i)->clone());
	}
	return result;
}
#pragma endregion
#pragma region sync
sync::sync() {
	prepared = vector<sPtr<model>>();
}
sync::sync(model* _modelTemplate) {
	this->modelTemplate = _modelTemplate;
	prepared = vector<sPtr<model>>();
}
void sync::fwd() {
	syncFwd(x, y, this);
}
void sync::modelWise(function<void(model*)> func) {
	func(this);
	syncModelWise(func, this);
}
sPtr<model> sync::clone() {
	sync* result = new sync();
	result->x = new cType(*x);
	result->y = new cType(*y);
	result->modelTemplate = modelTemplate;
	return result;
}
void sync::prep(int a) {
	for (int i = 0; i < a; i++) {
		prepared.push_back(modelTemplate->clone());
	}
}
void sync::unroll(int a) {

	// insure that the requested unroll size, 'a' will not cause there to be more instantiations of modelTemplate used than there are prepared
	assert(size() + a <= prepared.size());

	for (int i = 0; i < a; i++) {

		x->vVector.push_back(new cType({}));
		y->vVector.push_back(new cType({}));

		push_back(prepared.at(size()));

	}
}

syncBpg::syncBpg() {
	prepared = vector<sPtr<model>>();
}
syncBpg::syncBpg(model* _modelTemplate) {
	this->modelTemplate = _modelTemplate;
	prepared = vector<sPtr<model>>();
}
void syncBpg::fwd() {
	syncFwd(x, y, this);
}
void syncBpg::bwd() {
	syncBwd(yGrad, xGrad, this);
}
void syncBpg::modelWise(function<void(model*)> func) {
	func(this);
	syncModelWise(func, this);
}
sPtr<model> syncBpg::clone() {
	syncBpg* result = new syncBpg();
	result->x = new cType(*x);
	result->y = new cType(*y);
	result->xGrad = new cType(*xGrad);
	result->yGrad = new cType(*yGrad);
	result->modelTemplate = modelTemplate;
	return result;
}
void syncBpg::prep(int a) {
	for (int i = 0; i < a; i++) {
		prepared.push_back(modelTemplate->clone());
	}
}
void syncBpg::unroll(int a) {

	// insure that the requested unroll size, 'a' will not cause there to be more instantiations of modelTemplate used than there are prepared
	assert(size() + a <= prepared.size());

	for (int i = 0; i < a; i++) {

		x->vVector.push_back(new cType({}));
		y->vVector.push_back(new cType({}));
		xGrad->vVector.push_back(new cType({}));
		yGrad->vVector.push_back(new cType({}));

		push_back(prepared.at(size()));

	}
}
#pragma endregion
#pragma region lstmTS
lstmTS::lstmTS() {

	this->x = new cType({});
	this->y = new cType({});

	this->cTIn = new cType({});
	this->cTOut = new cType({});
	this->hTIn = new cType({});
	this->hTOut = new cType({});

}
lstmTS::lstmTS(int _units, sPtr<model> _aGate, sPtr<model> _bGate, sPtr<model> _cGate, sPtr<model> _dGate) {

	this->units = _units;
	this->aGate = _aGate;
	this->bGate = _bGate;
	this->cGate = _cGate;
	this->dGate = _dGate;

	this->x = new cType({});
	this->y = new cType({});

	this->cTIn = new cType({});
	this->cTOut = new cType({});
	this->hTIn = new cType({});
	this->hTOut = new cType({});

	for (int i = 0; i < units; i++) {

		this->cTIn->vVector.push_back(new cType());
		this->cTOut->vVector.push_back(new cType());
		this->hTIn->vVector.push_back(new cType());
		this->hTOut->vVector.push_back(new cType());

	}

}
void lstmTS::fwd() {
	lstmTSFwd(x, cTIn, hTIn, 
		y, cTOut, hTOut, 
		aGate, bGate, cGate, dGate);
}
void lstmTS::modelWise(function<void(model*)> func) {
	func(this);
	lstmTSModelWise(func, aGate, bGate, cGate, dGate);
}
sPtr<model> lstmTS::clone() {

	lstmTS* result = new lstmTS(units, aGate->clone(), bGate->clone(), cGate->clone(), dGate->clone());
	result->x = new cType(*x);
	result->y = new cType(*y);
	return result;

}

lstmTSBpg::lstmTSBpg() {

	this->x = new cType({});
	this->y = new cType({});

	this->cTIn = new cType({});
	this->cTOut = new cType({});
	this->hTIn = new cType({});
	this->hTOut = new cType({});

}
lstmTSBpg::lstmTSBpg(int _units, sPtr<model> _aGate, sPtr<model> _bGate, sPtr<model> _cGate, sPtr<model> _dGate) {

	this->units = _units;
	this->aGate = _aGate;
	this->bGate = _bGate;
	this->cGate = _cGate;
	this->dGate = _dGate;

	this->x = new cType({});
	this->y = new cType({});
	this->xGrad = new cType({});
	this->yGrad = new cType({});

	this->cTIn = new cType({});
	this->cTOut = new cType({});
	this->hTIn = new cType({});
	this->hTOut = new cType({});
	this->cTInGrad = new cType({});
	this->cTOutGrad = new cType({});
	this->hTInGrad = new cType({});
	this->hTOutGrad = new cType({});

	for (int i = 0; i < units; i++) {

		this->cTIn->vVector.push_back(new cType());
		this->cTOut->vVector.push_back(new cType());
		this->hTIn->vVector.push_back(new cType());
		this->hTOut->vVector.push_back(new cType());

		this->cTInGrad->vVector.push_back(new cType());
		this->cTOutGrad->vVector.push_back(new cType());
		this->hTInGrad->vVector.push_back(new cType());
		this->hTOutGrad->vVector.push_back(new cType());

	}

}
void lstmTSBpg::fwd() {
	lstmTSFwd(x, cTIn, hTIn,
		y, cTOut, hTOut,
		aGate, bGate, cGate, dGate);
}
void lstmTSBpg::bwd() {
	lstmTSBwd(units, cTOut, cTIn, yGrad,
		cTOutGrad, hTOutGrad, xGrad, cTInGrad, hTInGrad,
		aGate, bGate, cGate, dGate);
}
void lstmTSBpg::modelWise(function<void(model*)> func) {
	func(this);
	lstmTSModelWise(func, aGate, bGate, cGate, dGate);
}
sPtr<model> lstmTSBpg::clone() {

	lstmTSBpg* result = new lstmTSBpg(units, aGate->clone(), bGate->clone(), cGate->clone(), dGate->clone());
	result->x = new cType(*x);
	result->y = new cType(*y);
	result->xGrad = new cType(*xGrad);
	result->yGrad = new cType(*yGrad);
	return result;

}
#pragma endregion
#pragma region lstm
lstm::lstm() {

}
lstm::lstm(int _units) {

	this->units = _units;

	seq nlr = neuronLR(0.05);
	seq nth = neuronTh();
	seq nsm = neuronSm();

	sPtr<model> aGate = tnn({ 2 * units, units }, { &nlr, &nsm });
	sPtr<model> bGate = tnn({ 2 * units, units }, { &nlr, &nsm });
	sPtr<model> cGate = tnn({ 2 * units, units }, { &nlr, &nth });
	sPtr<model> dGate = tnn({ 2 * units, units }, { &nlr, &nsm });

	this->hTIn = new cType({});
	this->cTIn = new cType({});
	this->hTOut = new cType({});
	this->cTOut = new cType({});

	for (int i = 0; i < units; i++) {
		hTIn->vVector.push_back(new cType());
		cTIn->vVector.push_back(new cType());
		hTOut->vVector.push_back(new cType());
		cTOut->vVector.push_back(new cType());
	}

	this->lstmTSTemplate = new lstmTS(units, aGate, bGate, cGate, dGate);

}
lstm::lstm(int _units, sPtr<model> _aGate, sPtr<model> _bGate, sPtr<model> _cGate, sPtr<model> _dGate) {
	
	this->units = _units;

	this->hTIn = new cType({});
	this->cTIn = new cType({});
	this->hTOut = new cType({});
	this->cTOut = new cType({});

	for (int i = 0; i < units; i++) {
		hTIn->vVector.push_back(new cType());
		cTIn->vVector.push_back(new cType());
		hTOut->vVector.push_back(new cType());
		cTOut->vVector.push_back(new cType());
	}

	this->lstmTSTemplate = new lstmTS(units, _aGate, _bGate, _cGate, _dGate);

}
void lstm::fwd() {

	vector<sPtr<cType>>* xVec = &x->vVector;
	vector<sPtr<cType>>* yVec = &y->vVector;

	sPtr<cType> cT = cTIn;
	sPtr<cType> hT = hTIn;

	for (int i = 0; i < size(); i++) {

		lstmTS* l = (lstmTS*)at(i).get();
		l->x = xVec->at(i);
		l->cTIn = cT;
		l->hTIn = hT;
		l->fwd();
		yVec->at(i) = l->y;
		cT = l->cTOut;
		hT = l->hTOut;

	}

	*cTOut = *cT;
	*hTOut = *hT;

}
void lstm::modelWise(function<void(model*)> func) {

	func(this);
	lstmModelWise(func, lstmTSTemplate);

}
sPtr<model> lstm::clone() {

	lstm* result = new lstm();
	result->units = units;
	result->lstmTSTemplate = lstmTSTemplate->clone();
	return result;

}
void lstm::prep(int a) {
	for (int i = 0; i < a; i++) {
		prepared.push_back(lstmTSTemplate->clone());
	}
}
void lstm::unroll(int a) {
	for (int i = 0; i < a; i++) {
		push_back(prepared.at(size()));
		x->vVector.push_back(new cType{});
		y->vVector.push_back(new cType{});
	}
}

lstmBpg::lstmBpg() {

}
lstmBpg::lstmBpg(int _units) {

	this->units = _units;

	seqBpg nlr = neuronLRBpg(0.05);
	seqBpg nth = neuronThBpg();
	seqBpg nsm = neuronSmBpg();

	sPtr<model> aGate = tnnBpg({ 2 * units, units }, { &nlr, &nsm });
	sPtr<model> bGate = tnnBpg({ 2 * units, units }, { &nlr, &nsm });
	sPtr<model> cGate = tnnBpg({ 2 * units, units }, { &nlr, &nth });
	sPtr<model> dGate = tnnBpg({ 2 * units, units }, { &nlr, &nsm });

	this->hTIn = new cType({});
	this->cTIn = new cType({});
	this->hTOut = new cType({});
	this->cTOut = new cType({});

	this->hTInGrad = new cType({});
	this->cTInGrad = new cType({});
	this->hTOutGrad = new cType({});
	this->cTOutGrad = new cType({});

	for (int i = 0; i < units; i++) {
		hTIn->vVector.push_back(new cType());
		cTIn->vVector.push_back(new cType());
		hTOut->vVector.push_back(new cType());
		cTOut->vVector.push_back(new cType());

		hTInGrad->vVector.push_back(new cType());
		cTInGrad->vVector.push_back(new cType());
		hTOutGrad->vVector.push_back(new cType());
		cTOutGrad->vVector.push_back(new cType());
	}

	this->lstmTSTemplate = new lstmTSBpg(units, aGate, bGate, cGate, dGate);

}
lstmBpg::lstmBpg(int _units, sPtr<model> _aGate, sPtr<model> _bGate, sPtr<model> _cGate, sPtr<model> _dGate) {

	this->units = _units;

	this->hTIn = new cType({});
	this->cTIn = new cType({});
	this->hTOut = new cType({});
	this->cTOut = new cType({});

	this->hTInGrad = new cType({});
	this->cTInGrad = new cType({});
	this->hTOutGrad = new cType({});
	this->cTOutGrad = new cType({});

	for (int i = 0; i < units; i++) {
		hTIn->vVector.push_back(new cType());
		cTIn->vVector.push_back(new cType());
		hTOut->vVector.push_back(new cType());
		cTOut->vVector.push_back(new cType());

		hTInGrad->vVector.push_back(new cType());
		cTInGrad->vVector.push_back(new cType());
		hTOutGrad->vVector.push_back(new cType());
		cTOutGrad->vVector.push_back(new cType());
	}

	this->lstmTSTemplate = new lstmTSBpg(units, _aGate, _bGate, _cGate, _dGate);

}
void lstmBpg::fwd() {

	vector<sPtr<cType>>* xVec = &x->vVector;
	vector<sPtr<cType>>* yVec = &y->vVector;

	sPtr<cType> cT = cTIn;
	sPtr<cType> hT = hTIn;

	for (int i = 0; i < size(); i++) {

		lstmTSBpg* l = (lstmTSBpg*)at(i).get();
		l->x = xVec->at(i);
		l->cTIn = cT;
		l->hTIn = hT;
		l->fwd();
		yVec->at(i) = l->y;
		cT = l->cTOut;
		hT = l->hTOut;

	}

	*cTOut = *cT;
	*hTOut = *hT;

}
void lstmBpg::bwd() {

	vector<sPtr<cType>>* yGradVec = &yGrad->vVector;
	vector<sPtr<cType>>* xGradVec = &xGrad->vVector;

	sPtr<cType> cTGrad = cTOutGrad;
	sPtr<cType> hTGrad = hTOutGrad;

	for (int i = size() - 1; i >= 0; i--) {

		lstmTSBpg* l = (lstmTSBpg*)at(i).get();
		l->yGrad = yGradVec->at(i);
		l->cTOutGrad = cTGrad;
		l->hTOutGrad = hTGrad;
		l->bwd();
		xGradVec->at(i) = l->xGrad;
		cTGrad = l->cTInGrad;
		hTGrad = l->hTInGrad;

	}

	cTInGrad = cTGrad;
	hTInGrad = hTGrad;

}
void lstmBpg::modelWise(function<void(model*)> func) {
	
	func(this);
	lstmModelWise(func, lstmTSTemplate);

}
sPtr<model> lstmBpg::clone() {

	lstmBpg* result = new lstmBpg();
	result->units = units;
	result->lstmTSTemplate = lstmTSTemplate->clone();
	return result;

}
void lstmBpg::prep(int a) {
	for (int i = 0; i < a; i++) {
		prepared.push_back(lstmTSTemplate->clone());
	}
}
void lstmBpg::unroll(int a) {
	for (int i = 0; i < a; i++) {
		push_back(prepared.at(size()));
		x->vVector.push_back(new cType{});
		y->vVector.push_back(new cType{});
		xGrad->vVector.push_back(new cType{});
		yGrad->vVector.push_back(new cType{});
	}
}
#pragma endregion


#pragma endregion