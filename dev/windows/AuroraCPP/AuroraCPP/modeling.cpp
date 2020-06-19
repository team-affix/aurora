#pragma once
#include "modeling.h"

#pragma region functions

#pragma region external
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
	return new model(*this);
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
	return new bias(*this);
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
	return new biasBpg(*this);
}
#pragma endregion
#pragma region act
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
	return new act(*this);
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
	return new actBpg(*this);
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
	return new weight(*this);
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
	return new weightBpg(*this);
}
#pragma endregion
#pragma region wSet
wSet::wSet(int a) {
	x = new cType(0);
	y = new cType({});
	for (int i = 0; i < a; i++) {
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
	return new wSet(*this);
}
wSetBpg::wSetBpg(int a) {
	x = new cType(0);
	y = new cType({});
	xGrad = new cType(0);
	yGrad = new cType({});
	for (int i = 0; i < a; i++) {
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
	return new wSetBpg(*this);
}
#pragma endregion
#pragma region wJunc
wJunc::wJunc(int a, int b) {
	x = new cType({});
	y = new cType({});
	for (int i = 0; i < a; i++) {
		x->vVector.push_back(new cType(0));
		push_back(new wSet(b));
	}
	for (int i = 0; i < b; i++) {
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
	return new wJunc(*this);
}
wJuncBpg::wJuncBpg(int a, int b) {
	x = new cType({});
	y = new cType({});
	xGrad = new cType({});
	yGrad = new cType({});
	for (int i = 0; i < a; i++) {
		x->vVector.push_back(new cType(0));
		xGrad->vVector.push_back(new cType(0));
		push_back(new wSetBpg(b));
	}
	for (int i = 0; i < b; i++) {
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
	return new wJuncBpg(*this);
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
	return new seq(*this);
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
	seqBwd(x, y, this);
}
void seqBpg::modelWise(function<void(model*)> func) {
	func(this);
	seqModelWise(func, this);
}
sPtr<model> seqBpg::clone() {
	return new seqBpg(*this);
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
	return new layer(*this);
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
	return new layerBpg(*this);
}
#pragma endregion

#pragma endregion