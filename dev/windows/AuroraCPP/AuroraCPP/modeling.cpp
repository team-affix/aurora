#pragma once
#include "modeling.h"

#pragma region functions

// functions outside of classes

#pragma region external
seq* tnn(vector<int> npl, vector<ptr<model>> layerNeuronTemplates) {
	seq* result = new seq();
	for (int i = 0; i < npl.size() - 1; i++) {
		result->push_back(new layer(npl.at(i), layerNeuronTemplates.at(i)));
		result->push_back(new wJunc(npl.at(i), npl.at(i + 1)));
	}
	result->push_back(new layer(npl.back(), layerNeuronTemplates.back()));
	return result;
}
seq* tnn(vector<int> npl, ptr<model> neuronTemplate) {
	seq* result = new seq();
	for (int i = 0; i < npl.size() - 1; i++) {
		result->push_back(new layer(npl.at(i), neuronTemplate));
		result->push_back(new wJunc(npl.at(i), npl.at(i + 1)));
	}
	result->push_back(new layer(npl.back(), neuronTemplate));
	return result;
}
seq* cnn(ptr<model> _cnl, int _cnls) {
	
	seq* result = new seq();
	for (int i = 0; i < _cnls; i++) {
		result->push_back(_cnl->clone());
	}
	return result;
}
seq* cnn(int _a, int _b, int _cnls) {

	seq* result = new seq();
	for (int i = 0; i < _cnls; i++) {
		result->push_back(new cnl(_a, _b));
	}
	return result;

}
seq* cnn(vector<ptr<model>> _cnls) {

	seq* result = new seq(_cnls);
	return result;

}
seq* neuronSm() {

	// construct tanh neuron
	seq* nsm = new seq();
	nsm->push_back(new bias());
	nsm->push_back(new act(new actFuncSm()));
	return nsm;

}
seq* neuronTh() {

	// construct tanh neuron
	seq* nth = new seq();
	nth->push_back(new bias());
	nth->push_back(new act(new actFuncTh()));
	return nth;

}
seq* neuronLR(double m) {

	// construct tanh neuron
	seq* nlr = new seq();
	nlr->push_back(new bias());
	nlr->push_back(new act(new actFuncLR(m)));
	return nlr;

}
void extractParam(model* m, vector<ptr<param>*>& paramVecOutput) {
	if (bias* b = dynamic_cast<bias*>(m)) {
		b->prm = ptr<param>();
		paramVecOutput.push_back(&b->prm);
	}
	else if (bias* b = dynamic_cast<bias*>(m)) {
		b->prm = ptr<param>();
		paramVecOutput.push_back(&b->prm);
	}
	else if (weight* w = dynamic_cast<weight*>(m)) {
		w->prm = ptr<param>();
		paramVecOutput.push_back(&w->prm);
	}
	else if (weight* w = dynamic_cast<weight*>(m)) {
		w->prm = ptr<param>();
		paramVecOutput.push_back(&w->prm);
	}
}
void extractParams(model* m, vector<ptr<param>*>& paramVecOutput) {
	m->modelWise([&paramVecOutput](model* subModel) { extractParam(subModel, paramVecOutput); });
}
void extractParams(ptr<model> m, vector<ptr<param>*>& paramVecOutput) {
	extractParams(m.get(), paramVecOutput);
}
#pragma endregion

#pragma endregion

#pragma region definitions

// class function definitions

#pragma region model
model::model() {
	x = new cType(0);
	y = new cType(0);
	xGrad = new cType(0);
	yGrad = new cType(0);
}
void model::fwd() {
	*y = *x;
}
void model::bwd() {
	*xGrad = *yGrad;
}
void model::modelWise(function<void(model*)> func) {
	func(this);
}
ptr<model> model::clone() {
	return new model(*this);
}
#pragma endregion
#pragma region bias
bias::bias() {
	x = new cType(0);
	y = new cType(0);
	xGrad = new cType(0);
	yGrad = new cType(0);
}
void bias::fwd() {
	y->vDouble = x->vDouble + prm->state;
}
void bias::bwd() {
	paramSgd* p = (paramSgd*)prm.get();
	p->gradient += yGrad->vDouble;
	xGrad->vDouble = yGrad->vDouble;
}
ptr<model> bias::clone() {
	bias* result = new bias();
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
	xGrad = new cType(0);
	yGrad = new cType(0);
	this->af = _af;
}
act::act(ptr<actFunc> _af) {
	x = new cType(0);
	y = new cType(0);
	this->af = _af;
}
void act::fwd() {
	y->vDouble = af->eval(x->vDouble);
}
void act::bwd() {
	xGrad->vDouble = yGrad->vDouble * af->deriv(&x->vDouble, &y->vDouble);
}
ptr<model> act::clone() {
	act* result = new act();
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
	xGrad = new cType(0);
	yGrad = new cType(0);
}
void weight::fwd() {
	y->vDouble = x->vDouble * prm->state;
}
void weight::bwd() {
	paramSgd* p = (paramSgd*)prm.get();
	p->gradient += yGrad->vDouble * x->vDouble;
	xGrad->vDouble = yGrad->vDouble * prm->state;
}
ptr<model> weight::clone() {
	weight* result = new weight();
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
	xGrad = new cType(0);
	yGrad = new cType({});
	for (int i = 0; i < _a; i++) {
		y->vVector.push_back(new cType(0));
		yGrad->vVector.push_back(new cType(0));
		push_back(new weight());
	}
}
void wSet::fwd() {

	vector<ptr<cType>>* yVec = &y->vVector;
	for (int i = 0; i < size(); i++) {

		model* m = at(i).get();
		m->x->vDouble = x->vDouble;
		m->fwd();
		yVec->at(i)->vDouble = m->y->vDouble;

	}

}
void wSet::bwd() {

	// reset xGrad of weightSet to avoid overaccumulation
	xGrad->vDouble = 0;

	vector<ptr<cType>>* yGradVec = &yGrad->vVector;
	for (int i = 0; i < size(); i++) {

		model* m = at(i).get();
		m->yGrad->vDouble = yGradVec->at(i)->vDouble;
		m->bwd();
		xGrad->vDouble += m->xGrad->vDouble;

	}

}
void wSet::modelWise(function<void(model*)> func) {
	func(this);
	for (int i = 0; i < size(); i++) {
		at(i)->modelWise(func);
	}
}
ptr<model> wSet::clone() {
	wSet* result = new wSet();
	result->x = new cType();
	result->y = make1D(a);
	result->xGrad = new cType();
	result->yGrad = make1D(a);
	result->a = a;
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
	xGrad = new cType({});
	yGrad = new cType({});

	for (int i = 0; i < _a; i++) {
		x->vVector.push_back(new cType(0));
		xGrad->vVector.push_back(new cType(0));
		push_back(new wSet(b));
	}

	for (int i = 0; i < _b; i++) {
		y->vVector.push_back(new cType(0));
		yGrad->vVector.push_back(new cType(0));
	}

}
void wJunc::fwd() {

	// reset all values in vector to zero, to ensure that no overaccumulation occurs
	clear1D(y);
	vector<ptr<cType>>* xVec = &x->vVector;
	for (int i = 0; i < size(); i++) {

		model* m = at(i).get();
		m->x->vDouble = xVec->at(i)->vDouble;
		m->fwd();

		// accumulates output
		add1D(y, m->y, y);

	}

}
void wJunc::bwd() {

	vector<ptr<cType>>* xGradVec = &xGrad->vVector;
	for (int i = 0; i < size(); i++) {

		model* m = at(i).get();
		m->yGrad->vVector = yGrad->vVector;
		m->bwd();
		xGradVec->at(i)->vDouble = m->xGrad->vDouble;

	}

}
void wJunc::modelWise(function<void(model*)> func) {
	func(this);
	for (int i = 0; i < size(); i++) {
		at(i)->modelWise(func);
	}
}
ptr<model> wJunc::clone() {
	wJunc* result = new wJunc();
	result->x = make1D(a);
	result->y = make1D(b);
	result->xGrad = make1D(a);
	result->yGrad = make1D(b);
	result->a = a;
	result->b = b;
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
	xGrad = new cType(0);
	yGrad = new cType(0);

}
seq::seq(vector<ptr<model>>& _models) {
	
	for (int i = 0; i < _models.size(); i++) {
		push_back(_models.at(i));
	}

}
seq::seq(initializer_list<ptr<model>> _models) {

	x = new cType(0);
	y = new cType(0);
	xGrad = new cType(0);
	yGrad = new cType(0);

	assign(_models);

}
void seq::fwd() {

	ptr<cType> currentInput = x;
	for (int i = 0; i < size(); i++) {

		model* m = at(i).get();
		m->x = currentInput;
		m->fwd();
		currentInput = m->y;

	}
	*y = *currentInput;

}
void seq::bwd() {

	ptr<cType> currentGradient = yGrad;
	for (int i = size() - 1; i >= 0; i--) {

		model* m = at(i).get();
		m->yGrad = currentGradient;
		m->bwd();
		currentGradient = m->xGrad;

	}
	*xGrad = *currentGradient;

}
void seq::modelWise(function<void(model*)> func) {
	func(this);
	for (int i = 0; i < size(); i++) {
		at(i)->modelWise(func);
	}
}
ptr<model> seq::clone() {
	seq* result = new seq();
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

	// initialized the x, y, xGrad, yGrad cTypes as vectors to insure that all members of the cType are initialized
	x = new cType({});
	y = new cType({});
	xGrad = new cType({});
	yGrad = new cType({});

}
layer::layer(int a, model* modelTemplate) {

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
layer::layer(int a, ptr<model> modelTemplate) {

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
layer::layer(initializer_list<ptr<model>> _models) {

	x = new cType({});
	y = new cType({});
	xGrad = new cType({});
	yGrad = new cType({});

	assign(_models);

	for (int i = 0; i < size(); i++) {

		// initialize the x, y, xGrad, yGrad cTypes as vectors to insure that all members of the cType are initialized
		x->vVector.push_back(new cType({}));
		y->vVector.push_back(new cType({}));
		xGrad->vVector.push_back(new cType({}));
		yGrad->vVector.push_back(new cType({}));

	}

}
void layer::fwd() {

	vector<ptr<cType>>* xVec = &x->vVector;
	vector<ptr<cType>>* yVec = &y->vVector;
	for (int i = 0; i < size(); i++) {

		model* m = at(i).get();
		m->x = xVec->at(i);
		m->fwd();
		yVec->at(i) = m->y;

	}

}
void layer::bwd() {

	vector<ptr<cType>>* xGradVec = &xGrad->vVector;
	vector<ptr<cType>>* yGradVec = &yGrad->vVector;
	for (int i = 0; i < size(); i++) {

		model* m = at(i).get();
		m->yGrad = yGradVec->at(i);
		m->bwd();
		xGradVec->at(i) = m->xGrad;

	}

}
void layer::modelWise(function<void(model*)> func) {
	func(this);
	for (int i = 0; i < size(); i++) {
		at(i)->modelWise(func);
	}
}
ptr<model> layer::clone() {
	layer* result = new layer();
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
	prepared = vector<ptr<model>>();
}
sync::sync(model* _modelTemplate) {
	this->modelTemplate = _modelTemplate;
	prepared = vector<ptr<model>>();
}
sync::sync(ptr<model> _modelTemplate) {
	this->modelTemplate = _modelTemplate;
	prepared = vector<ptr<model>>();
}
void sync::fwd() {
	incFwd(size());
}
void sync::incFwd(int a) {

	vector<ptr<cType>>* xVec = &x->vVector;
	vector<ptr<cType>>* yVec = &y->vVector;

	for (int j = 0; j < a; j++) {

		model* m = at(index).get();
		m->x = xVec->at(index);
		m->fwd();
		yVec->at(index) = m->y;
		index++;

	}

}
void sync::bwd() {
	incBwd(index);
}
void sync::incBwd(int a) {

	vector<ptr<cType>>* xGradVec = &xGrad->vVector;
	vector<ptr<cType>>* yGradVec = &yGrad->vVector;

	for (int j = 0; j < a; j++) {

		index--;
		model* m = at(index).get();
		m->yGrad = yGradVec->at(index);
		m->bwd();
		xGradVec->at(index) = m->xGrad;

	}

}
void sync::modelWise(function<void(model*)> func) {
	func(this);
	modelTemplate->modelWise(func);
}
ptr<model> sync::clone() {
	sync* result = new sync();
	result->x = new cType(*x);
	result->y = new cType(*y);
	result->xGrad = new cType(*xGrad);
	result->yGrad = new cType(*yGrad);
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
		xGrad->vVector.push_back(new cType({}));
		yGrad->vVector.push_back(new cType({}));

		push_back(prepared.at(size()));

	}
}
void sync::clear() {
	vector<ptr<model>>::clear();
	x->vVector.clear();
	y->vVector.clear();
	xGrad->vVector.clear();
	yGrad->vVector.clear();
	index = 0;
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
lstmTS::lstmTS(int _units, ptr<model> _aGate, ptr<model> _bGate, ptr<model> _cGate, ptr<model> _dGate) {

	this->units = _units;
	this->aGate = _aGate;
	this->bGate = _bGate;
	this->cGate = _cGate;
	this->dGate = _dGate;

	this->x = make1D(units);
	this->y = make1D(units);
	this->xGrad = make1D(units);
	this->yGrad = make1D(units);

	// cast all gates to the models that they are
	model* aGateBpg = (model*)aGate.get();
	model* bGateBpg = (model*)bGate.get();
	model* cGateBpg = (model*)cGate.get();
	model* dGateBpg = (model*)dGate.get();

	aGateBpg->x = make1D(2 * units);
	aGateBpg->y = make1D(units);
	bGateBpg->x = make1D(2 * units);
	bGateBpg->y = make1D(units);
	cGateBpg->x = make1D(2 * units);
	cGateBpg->y = make1D(units);
	dGateBpg->x = make1D(2 * units);
	dGateBpg->y = make1D(units);
	aGateBpg->yGrad = make1D(units);
	aGateBpg->xGrad = make1D(2 * units);
	bGateBpg->yGrad = make1D(units);
	bGateBpg->xGrad = make1D(2 * units);
	cGateBpg->yGrad = make1D(units);
	cGateBpg->xGrad = make1D(2 * units);
	dGateBpg->yGrad = make1D(units);
	dGateBpg->xGrad = make1D(2 * units);

	this->cTIn = make1D(units);
	this->cTOut = make1D(units);
	this->hTIn = make1D(units);
	this->hTOut = make1D(units);

	this->cTInGrad = make1D(units);
	this->cTOutGrad = make1D(units);
	this->hTInGrad = make1D(units);
	this->hTOutGrad = make1D(units);

	this->comp_LenUnits = make2D(3, units);
	this->comp_Len2Units = make2D(3, 2 * units);

}
void lstmTS::fwd() {

	// import cTypes used for computation output that are of length: units
	vector<ptr<cType>>* compVec_LenUnits = &comp_LenUnits->vVector;
	ptr<cType> cT = compVec_LenUnits->at(0);
	ptr<cType> bYTimescY = compVec_LenUnits->at(1);

	// import cTypes used for computation output that are of length: 2 * units
	vector<ptr<cType>>* compVec_Len2Units = &comp_Len2Units->vVector;
	ptr<cType> xConcatHTIn = compVec_Len2Units->at(0);

	concat(x, hTIn, xConcatHTIn);

	copy1D(xConcatHTIn, aGate->x);
	copy1D(xConcatHTIn, bGate->x);
	copy1D(xConcatHTIn, cGate->x);
	copy1D(xConcatHTIn, dGate->x);

	aGate->fwd();
	bGate->fwd();
	cGate->fwd();
	dGate->fwd();

	mult1D(aGate->y, cTIn, cT);
	mult1D(bGate->y, cGate->y, bYTimescY);
	add1D(cT, bYTimescY, cTOut);
	mult1D(cTOut, dGate->y, hTOut);
	copy1D(hTOut, y);

}
void lstmTS::bwd() {

	// import cTypes used for computation output that are of length: units
	vector<ptr<cType>>* compVec_LenUnits = &comp_LenUnits->vVector;
	ptr<cType> hTGrad = compVec_LenUnits->at(0);
	ptr<cType> dYTimeshTGrad = compVec_LenUnits->at(1);
	ptr<cType> cTGrad = compVec_LenUnits->at(2);

	// import cTypes used for computation output that are of length: 2 * units
	vector<ptr<cType>>* compVec_Len2Units = &comp_Len2Units->vVector;
	ptr<cType> xGradSum1 = compVec_Len2Units->at(0);
	ptr<cType> xGradSum2 = compVec_Len2Units->at(1);
	ptr<cType> xGradSum3 = compVec_Len2Units->at(2);

	// cast all gates to the models that they are
	model* aGateBpg = (model*)aGate.get();
	model* bGateBpg = (model*)bGate.get();
	model* cGateBpg = (model*)cGate.get();
	model* dGateBpg = (model*)dGate.get();

	// calculate major gradient tracks
	add1D(hTOutGrad, yGrad, hTGrad);
	mult1D(dGateBpg->y, hTGrad, dYTimeshTGrad);
	add1D(cTOutGrad, dYTimeshTGrad, cTGrad);

	// calculate each gate's output gradient
	mult1D(hTGrad, cTOut, dGateBpg->yGrad);
	mult1D(bGateBpg->y, cTGrad, cGateBpg->yGrad);
	mult1D(cGateBpg->y, cTGrad, bGateBpg->yGrad);
	mult1D(cTGrad, cTIn, aGateBpg->yGrad);

	mult1D(aGateBpg->y, cTGrad, cTInGrad);

	// carry each gates' gradient backward
	aGateBpg->bwd();
	bGateBpg->bwd();
	cGateBpg->bwd();
	dGateBpg->bwd();

	add1D(aGateBpg->xGrad, bGateBpg->xGrad, xGradSum1);
	add1D(cGateBpg->xGrad, dGateBpg->xGrad, xGradSum2);
	add1D(xGradSum1, xGradSum2, xGradSum3);

	copy1D(xGradSum3, xGrad, 0, units, 0);
	copy1D(xGradSum3, hTInGrad, units, units, 0);

}
void lstmTS::modelWise(function<void(model*)> func) {
	func(this);
	aGate->modelWise(func);
	bGate->modelWise(func);
	cGate->modelWise(func);
	dGate->modelWise(func);
}
ptr<model> lstmTS::clone() {

	lstmTS* result = new lstmTS(units, aGate->clone(), bGate->clone(), cGate->clone(), dGate->clone());
	return result;

}
#pragma endregion
#pragma region lstm
lstm::lstm() {

}
lstm::lstm(int _units) {

	this->units = _units;

	ptr<model> nlr = neuronLR(0.3);
	ptr<model> nth = neuronTh();
	ptr<model> nsm = neuronSm();

	ptr<model> aGate = tnn({ 2 * units, units }, { nlr, nsm });
	ptr<model> bGate = tnn({ 2 * units, units }, { nlr, nsm });
	ptr<model> cGate = tnn({ 2 * units, units }, { nlr, nth });
	ptr<model> dGate = tnn({ 2 * units, units }, { nlr, nsm });

	this->hTIn = make1D(units);
	this->cTIn = make1D(units);
	this->hTOut = make1D(units);
	this->cTOut = make1D(units);

	this->hTInGrad = make1D(units);
	this->cTInGrad = make1D(units);
	this->hTOutGrad = make1D(units);
	this->cTOutGrad = make1D(units);

	this->lstmTSTemplate = new lstmTS(units, aGate, bGate, cGate, dGate);

}
lstm::lstm(int _units, ptr<model> _aGate, ptr<model> _bGate, ptr<model> _cGate, ptr<model> _dGate) {

	this->units = _units;

	this->hTIn = make1D(units);
	this->cTIn = make1D(units);
	this->hTOut = make1D(units);
	this->cTOut = make1D(units);

	this->hTInGrad = make1D(units);
	this->cTInGrad = make1D(units);
	this->hTOutGrad = make1D(units);
	this->cTOutGrad = make1D(units);

	this->lstmTSTemplate = new lstmTS(units, _aGate, _bGate, _cGate, _dGate);

}
void lstm::fwd() {

	incFwd(size() - index);

}
void lstm::incFwd(int a) {

	vector<ptr<cType>>* xVec = &x->vVector;
	vector<ptr<cType>>* yVec = &y->vVector;

	ptr<cType> cT = make1D(units);
	ptr<cType> hT = make1D(units);

	if (index == 0) {
		copy1D(cTIn, cT);
		copy1D(hTIn, hT);
	}
	else {
		copy1D(cTOut, cT);
		copy1D(hTOut, hT);
	}

	for (int j = 0; j < a; j++) {

		lstmTS* l = (lstmTS*)at(index).get();
		l->x = xVec->at(index);
		l->cTIn = cT;
		l->hTIn = hT;
		l->fwd();
		yVec->at(index) = l->y;
		cT = l->cTOut;
		hT = l->hTOut;

		index++;

	}

	copy1D(cT, cTOut);
	copy1D(hT, hTOut);
	/*cTOut->vVector = cT->vVector;
	hTOut->vVector = hT->vVector;*/

}
void lstm::bwd() {

	incBwd(index);

}
void lstm::incBwd(int a) {

	vector<ptr<cType>>* yGradVec = &yGrad->vVector;
	vector<ptr<cType>>* xGradVec = &xGrad->vVector;

	ptr<cType> cTGrad = make1D(units);
	ptr<cType> hTGrad = make1D(units);

	if (index == size()) {
		copy1D(cTOutGrad, cTGrad);
		copy1D(hTOutGrad, hTGrad);
	}
	else {
		copy1D(cTInGrad, cTGrad);
		copy1D(hTInGrad, hTGrad);
	}

	for (int j = 0; j < a; j++) {

		index--;

		lstmTS* l = (lstmTS*)at(index).get();
		l->yGrad = yGradVec->at(index);
		l->cTOutGrad = cTGrad;
		l->hTOutGrad = hTGrad;
		l->bwd();
		xGradVec->at(index) = l->xGrad;
		cTGrad = l->cTInGrad;
		hTGrad = l->hTInGrad;

	}
	
	copy1D(cTGrad, cTInGrad);
	copy1D(hTGrad, hTInGrad);

}
void lstm::modelWise(function<void(model*)> func) {
	
	func(this);
	lstmTSTemplate->modelWise(func);

}
ptr<model> lstm::clone() {

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
		xGrad->vVector.push_back(new cType{});
		yGrad->vVector.push_back(new cType{});
	}
}
void lstm::clear() {
	vector<ptr<model>>::clear();
	x->vVector.clear();
	y->vVector.clear();
	xGrad->vVector.clear();
	yGrad->vVector.clear();
	clear1D(cTIn);
	clear1D(hTIn);
	clear1D(cTOut);
	clear1D(hTOut);
	clear1D(cTInGrad);
	clear1D(hTInGrad);
	clear1D(cTOutGrad);
	clear1D(hTOutGrad);
	index = 0;
}
#pragma endregion
#pragma region muTS
muTS::muTS() {

}
muTS::muTS(int _xUnits, int _cTUnits, int _hTUnits, ptr<model> _gate) {

	this->xUnits = _xUnits;
	this->cTUnits = _cTUnits;
	this->hTUnits = _hTUnits;
	this->gate = _gate;

	this->x = make1D(xUnits);
	this->y = make1D(hTUnits);
	this->xGrad = make1D(xUnits);
	this->yGrad = make1D(hTUnits);

	model* gateBpg = (model*)gate.get();

	gateBpg->x = make1D(xUnits + cTUnits + hTUnits);
	gateBpg->y = make1D(cTUnits + hTUnits);
	gateBpg->xGrad = make1D(xUnits + cTUnits + hTUnits);
	gateBpg->yGrad = make1D(cTUnits + hTUnits);

	this->cTIn = make1D(cTUnits);
	this->cTOut = make1D(cTUnits);
	this->hTIn = make1D(hTUnits);
	this->hTOut = make1D(hTUnits);

	this->cTInGrad = make1D(cTUnits);
	this->cTOutGrad = make1D(cTUnits);
	this->hTInGrad = make1D(hTUnits);
	this->hTOutGrad = make1D(hTUnits);

	this->comp_LenCTUnits = make2D(1, cTUnits);
	this->comp_LenHTUnits = make2D(1, hTUnits);

}
void muTS::fwd() {

	// import compuation cTypes
	vector<ptr<cType>>* comp_LenCTUnitsVec = &comp_LenCTUnits->vVector;
	ptr<cType> cTAddOperand = comp_LenCTUnitsVec->at(0);

	concat({ x, cTIn, hTIn }, gate->x);
	gate->fwd();
	copy1D(gate->y, cTAddOperand, 0, cTUnits, 0);
	add1D(cTIn, cTAddOperand, cTOut);
	copy1D(gate->y, hTOut, cTUnits, hTUnits, 0);
	copy1D(hTOut, y);

}
void muTS::bwd() {

	// import compuation cTypes
	vector<ptr<cType>>* comp_LenCTUnitsVec = &comp_LenCTUnits->vVector;
	ptr<cType> cTCopyOperand = comp_LenCTUnitsVec->at(0);

	vector<ptr<cType>>* comp_LenHTUnitsVec = &comp_LenHTUnits->vVector;
	ptr<cType> hTAddOperand = comp_LenHTUnitsVec->at(0);

	model* gateBpg = (model*)gate.get();

	add1D(yGrad, hTOutGrad, hTAddOperand);
	copy1D(cTOutGrad, gateBpg->yGrad, 0, cTUnits, 0);
	copy1D(hTAddOperand, gateBpg->yGrad, 0, hTUnits, cTUnits);
	gateBpg->bwd();
	copy1D(gateBpg->xGrad, cTCopyOperand, xUnits, cTUnits, 0);
	add1D(cTCopyOperand, cTOutGrad, cTInGrad);
	copy1D(gateBpg->xGrad, xGrad, 0, xUnits, 0);
	copy1D(gateBpg->xGrad, hTInGrad, xUnits + cTUnits, hTUnits, 0);

}
void muTS::modelWise(function<void(model*)> func) {

	func(this);
	gate->modelWise(func);

}
ptr<model> muTS::clone() {

	muTS* result = new muTS(xUnits, cTUnits, hTUnits, gate->clone());
	return result;

}
#pragma endregion
#pragma region mu
mu::mu() {

}
mu::mu(int _xUnits, int _cTUnits, int _hTUnits) {

	this->xUnits = _xUnits;
	this->cTUnits = _cTUnits;
	this->hTUnits = _hTUnits;

	this->cTIn = make1D(cTUnits);
	this->cTOut = make1D(cTUnits);
	this->hTIn = make1D(hTUnits);
	this->hTOut = make1D(hTUnits);

	this->cTInGrad = make1D(cTUnits);
	this->cTOutGrad = make1D(cTUnits);
	this->hTInGrad = make1D(hTUnits);
	this->hTOutGrad = make1D(hTUnits);

	ptr<model> nlr = neuronLR(0.3);

	ptr<model> gateTemplate = tnn({ xUnits + cTUnits + hTUnits, cTUnits + hTUnits }, { nlr, nlr });
	muTSTemplate = new muTS(xUnits, cTUnits, hTUnits, gateTemplate);

}
// Input layer size: xUnits + cTUnits + hTUnits. Output layer size: cTUnits + hTUnits
mu::mu(int _xUnits, int _cTUnits, int _hTUnits, ptr<model> _gate) {

	this->xUnits = _xUnits;
	this->cTUnits = _cTUnits;
	this->hTUnits = _hTUnits;

	this->cTIn = make1D(cTUnits);
	this->cTOut = make1D(cTUnits);
	this->hTIn = make1D(hTUnits);
	this->hTOut = make1D(hTUnits);

	this->cTInGrad = make1D(cTUnits);
	this->cTOutGrad = make1D(cTUnits);
	this->hTInGrad = make1D(hTUnits);
	this->hTOutGrad = make1D(hTUnits);

	muTSTemplate = new muTS(xUnits, cTUnits, hTUnits, _gate);

}
void mu::fwd() {

	incFwd(size() - index);

}
void mu::incFwd(int a) {

	vector<ptr<cType>>* xVec = &x->vVector;
	vector<ptr<cType>>* yVec = &y->vVector;

	ptr<cType> cT = make1D(cTUnits);
	ptr<cType> hT = make1D(hTUnits);

	// if the current carry index is 0, it means that none of the timesteps have been carried forward yet
	if (index == 0) {
		copy1D(cTIn, cT);
		copy1D(hTIn, hT);
	}
	else {
		copy1D(cTOut, cT);
		copy1D(hTOut, hT);
	}

	for (int j = 0; j < a; j++) {

		muTS* m = (muTS*)at(index).get();
		m->x = xVec->at(index);
		m->cTIn = cT;
		m->hTIn = hT;
		m->fwd();
		yVec->at(index) = m->y;
		cT = m->cTOut;
		hT = m->hTOut;

		index++;

	}


	copy1D(cT, cTOut);
	copy1D(hT, hTOut);

}
void mu::bwd() {

	incBwd(index);

}
void mu::incBwd(int a) {

	vector<ptr<cType>>* yGradVec = &yGrad->vVector;
	vector<ptr<cType>>* xGradVec = &xGrad->vVector;

	ptr<cType> cTGrad = make1D(cTUnits);
	ptr<cType> hTGrad = make1D(hTUnits);

	// if the current carry index is size(), it means that all of the timesteps have been carried forward.
	if (index == size()) {
		copy1D(cTOutGrad, cTGrad);
		copy1D(hTOutGrad, hTGrad);
	}
	else {
		copy1D(cTInGrad, cTGrad);
		copy1D(hTInGrad, hTGrad);
	}

	for (int j = 0; j < a; j++) {

		index--;

		muTS* m = (muTS*)at(index).get();
		m->yGrad = yGradVec->at(index);
		m->cTOutGrad = cTGrad;
		m->hTOutGrad = hTGrad;
		m->bwd();
		xGradVec->at(index) = m->xGrad;
		cTGrad = m->cTInGrad;
		hTGrad= m->hTInGrad;

	}

	copy1D(cTGrad, cTInGrad);
	copy1D(hTGrad, hTInGrad);

}
void mu::modelWise(function<void(model*)> func) {

	func(this);
	muTSTemplate->modelWise(func);

}
ptr<model> mu::clone() {

	muTS* castedTemplate = (muTS*)muTSTemplate.get();
	mu* result = new mu(xUnits, cTUnits, hTUnits, castedTemplate->gate->clone());
	return result;

}
void mu::prep(int a) {
	for (int i = 0; i < a; i++) {
		prepared.push_back(muTSTemplate->clone());
	}
}
void mu::unroll(int a) {
	for (int i = 0; i < a; i++) {
		push_back(prepared.at(size()));
		x->vVector.push_back(new cType{});
		y->vVector.push_back(new cType{});
		xGrad->vVector.push_back(new cType{});
		yGrad->vVector.push_back(new cType{});
	}
}
void mu::clear() {
	vector<ptr<model>>::clear();
	x->vVector.clear();
	y->vVector.clear();
	clear1D(cTIn);
	clear1D(hTIn);
	clear1D(cTOut);
	clear1D(hTOut);
	clear1D(cTInGrad);
	clear1D(hTInGrad);
	clear1D(cTOutGrad);
	clear1D(hTOutGrad);
	index = 0;
}
#pragma endregion
#pragma region attTS
attTS::attTS() {

}
attTS::attTS(int _xUnits, int _hTUnits) {

	// Make the xUnits and hTUnits publicly accessable
	this->xUnits = _xUnits;
	this->hTUnits = _hTUnits;

	// Initialize x, y, and hT vectors
	this->hTIn = make1D(hTUnits);
	this->hTInGrad = make1D(hTUnits);
	this->y = make1D(xUnits);
	this->yGrad = make1D(xUnits);

	// Initialize the prepared vector
	this->prepared = vector<ptr<model>>();

	// Instantiate neurons that will be used in attention template TNN
	ptr<model> nlr = neuronLR(0.3);
	ptr<model> nsm = neuronSm();

	// Initialize the attention timestep template model to be a tnn
	seq* seqTemp = tnn({ xUnits + hTUnits, xUnits }, { nlr, nsm });
	seqTemp->x = make1D(xUnits + hTUnits);
	seqTemp->y = make1D(xUnits);
	seqTemp->xGrad = make1D(xUnits + hTUnits);
	seqTemp->yGrad = make1D(xUnits);
	this->seqTemplate = seqTemp;

	// Initialize the template's x and y vectors
	seqTemplate->x = make1D(xUnits + hTUnits);
	seqTemplate->y = make1D(xUnits);

	// Initialize the computation result cType
	comp_LenXUnits = make2D(2, xUnits);
	comp_LenHTUnits = make2D(1, hTUnits);

}
attTS::attTS(int _xUnits, int _hTUnits, ptr<model> _seqTemplate) {

	// Make the xUnits and hTUnits publicly accessable
	this->xUnits = _xUnits;
	this->hTUnits = _hTUnits;

	// Initialize x, y, and hT vectors
	this->hTIn = make1D(hTUnits);
	this->hTInGrad = make1D(hTUnits);
	this->y = make1D(xUnits);
	this->yGrad = make1D(xUnits);

	// Initialize the prepared vector
	this->prepared = vector<ptr<model>>();

	// Initialize the attention timestep template model to be a tnn
	seq* seqTemp = (seq*)_seqTemplate.get();
	seqTemp->x = make1D(xUnits + hTUnits);
	seqTemp->y = make1D(xUnits);
	seqTemp->xGrad = make1D(xUnits + hTUnits);
	seqTemp->yGrad = make1D(xUnits);
	this->seqTemplate = seqTemp;

	// Initialize the computation result cType
	comp_LenXUnits = make2D(2, xUnits);
	comp_LenHTUnits = make2D(1, hTUnits);

}
void attTS::fwd() {

	incFwd(size());

}
void attTS::incFwd(int a) {

	vector<ptr<cType>>* comp_LenXUnitsVec = &comp_LenXUnits->vVector;
	ptr<cType> x_yproduct = comp_LenXUnitsVec->at(0);

	// pull in references to input matrices to save compute
	vector<ptr<cType>>* xVec = &x->vVector;

	clear1D(y);

	for (int j = 0; j < a; j++) {

		model* m = at(index).get();
		// set input to the model
		concat(xVec->at(index), hTIn, m->x);
		m->fwd();
		mult1D(m->y, xVec->at(index), x_yproduct);
		add1D(y, x_yproduct, y);

		index++;

	}

}
void attTS::bwd() {

	incBwd(index);

}
void attTS::incBwd(int a) {

	vector<ptr<cType>>* comp_LenXUnitsVec = &comp_LenXUnits->vVector;
	ptr<cType> xGrad_copyResult = comp_LenXUnitsVec->at(0);
	ptr<cType> y_yGradproduct = comp_LenXUnitsVec->at(1);

	vector<ptr<cType>>* comp_LenHTUnitsVec = &comp_LenHTUnits->vVector;
	ptr<cType> hTInGrad_copyResult = comp_LenHTUnitsVec->at(0);

	// pull in references to input matrices to save compute
	vector<ptr<cType>>* xGradVec = &xGrad->vVector;
	vector<ptr<cType>>* xVec = &x->vVector;

	clear1D(hTInGrad);

	for (int j = 0; j < a; j++) {

		index--;
		model* m = at(index).get();

		mult1D(yGrad, xVec->at(index), m->yGrad);
		m->bwd();
		copy1D(m->xGrad, xGrad_copyResult, 0, xUnits, 0);
		copy1D(m->xGrad, hTInGrad_copyResult, xUnits, hTUnits, 0);
		add1D(hTInGrad, hTInGrad_copyResult, hTInGrad);
		mult1D(yGrad, m->y, y_yGradproduct);
		add1D(y_yGradproduct, xGrad_copyResult, xGradVec->at(index));

	}
}
void attTS::modelWise(function<void(model*)> func) {

	func(this);
	seqTemplate->modelWise(func);

}
ptr<model> attTS::clone() {

	attTS* result = new attTS(xUnits, hTUnits, seqTemplate);
	return result;

}
void attTS::prep(int a) {

	for (int i = 0; i < a; i++) {
		prepared.push_back(seqTemplate->clone());
	}

}
void attTS::unroll(int a) {

	for (int i = 0; i < a; i++) {
		push_back(prepared.at(size()));
		x->vVector.push_back(make1D(xUnits));
		xGrad->vVector.push_back(make1D(xUnits));
	}

}
void attTS::clear() {

	vector<ptr<model>>::clear();
	x->vVector.clear();
	index = 0;

}
#pragma endregion
#pragma region att
att::att() {

}
att::att(int _xUnits, int _hTUnits) {

	// Make the xUnits and hTUnits publicly accessable
	this->xUnits = _xUnits;
	this->hTUnits = _hTUnits;

	this->x = new cType{};
	this->hTIn = new cType{};

	this->xGrad = new cType{};
	this->hTInGrad = new cType{};

	// Initialize the prepared vector
	this->prepared = vector<ptr<model>>();

	// Initialize template model
	attTS* attTSTemplateBpg = new attTS(xUnits, hTUnits);
	attTSTemplateBpg->y = make1D(xUnits);
	attTSTemplateBpg->yGrad = make1D(xUnits);
	this->attTSTemplate = attTSTemplateBpg;

}
att::att(int _xUnits, int _hTUnits, ptr<model> _attTSTemplate) {

	// Make the xUnits and hTUnits publicly accessable
	this->xUnits = _xUnits;
	this->hTUnits = _hTUnits;

	this->x = new cType{};
	this->hTIn = new cType{};

	this->xGrad = new cType{};
	this->hTInGrad = new cType{};

	// Initialize the prepared vector
	this->prepared = vector<ptr<model>>();

	// Initialize template model
	attTS* attTSTemplateBpg = (attTS*)_attTSTemplate.get();
	attTSTemplateBpg->y = make1D(xUnits);
	attTSTemplateBpg->yGrad = make1D(xUnits);
	this->attTSTemplate = attTSTemplateBpg;

}
void att::fwd() {

	// pull in reference to vector to save compute
	vector<ptr<cType>>* yVec = &y->vVector;
	// In reality, this vector is a vector of vectors, so a matrix.
	vector<ptr<cType>>* hTInMat = &hTIn->vVector;

	attTS* a = (attTS*)at(index).get();
	a->x = x;
	a->hTIn = hTInMat->at(index);
	a->fwd();
	yVec->at(index) = a->y;

	index++;

}
void att::incFwd(int _a) {

	// pull in reference to vector to save compute
	vector<ptr<cType>>* yVec = &y->vVector;
	// In reality, this vector is a vector of vectors, so a matrix.
	vector<ptr<cType>>* hTInMat = &hTIn->vVector;

	for (int j = 0; j < _a; j++) {

		attTS* a = (attTS*)at(index).get();
		a->x = x;
		a->hTIn = hTInMat->at(index);
		a->fwd();
		yVec->at(index) = a->y;

		index++;

	}

}
void att::bwd() {

	incBwd(index);

}
void att::incBwd(int _a) {

	// pull in reference to vector to save compute
	vector<ptr<cType>>* yGradVec = &yGrad->vVector;
	vector<ptr<cType>>* xGradVec = &xGrad->vVector;
	// In reality, this vector is a vector of vectors, so a matrix.
	vector<ptr<cType>>* hTInGradMat = &hTInGrad->vVector;

	for (int j = 0; j < _a; j++) {

		index--;

		attTS* a = (attTS*)at(index).get();
		a->yGrad = yGradVec->at(index);
		a->bwd();
		add2D(xGrad, a->xGrad, xGrad);
		hTInGradMat->at(index) = a->hTInGrad;

	}

}
void att::modelWise(function<void(model*)> func) {

	func(this);
	attTSTemplate->modelWise(func);

}
ptr<model> att::clone() {

	att* result = new att(xUnits, hTUnits, attTSTemplate->clone());
	return result;

}
void att::prep(int a) {

	for (int i = 0; i < a; i++) {
		prepared.push_back(attTSTemplate->clone());
	}

}
void att::prep(int a, int b) {

	for (int i = 0; i < a; i++) {
		ptr<model> m = attTSTemplate->clone();
		attTS* att = (attTS*)m.get();
		att->prep(b);
		prepared.push_back(m);
	}

}
void att::unroll(int a) {

	for (int i = 0; i < a; i++) {

		push_back(prepared.at(size()));
		x->vVector.push_back(make1D(xUnits));
		xGrad->vVector.push_back(make1D(xUnits));
		y->vVector.push_back(new cType{});
		yGrad->vVector.push_back(new cType{});
		hTIn->vVector.push_back(new cType{});
		hTInGrad->vVector.push_back(new cType{});

	}

}
void att::unroll(int a, int b) {

	for (int i = 0; i < a; i++) {

		int s = size();
		ptr<model> m = prepared.at(size());
		attTS* att = (attTS*)m.get();
		att->unroll(b);
		push_back(m);
		x->vVector.push_back(make1D(xUnits));
		xGrad->vVector.push_back(make1D(xUnits));
		y->vVector.push_back(new cType{});
		yGrad->vVector.push_back(new cType{});
		hTIn->vVector.push_back(new cType{});
		hTInGrad->vVector.push_back(new cType{});
		// x's vVector is not pushed back to because it is completely populated by the user before any timestep is carried forward

	}

}
void att::clear() {

	vector<ptr<model>>::clear();
	x->vVector.clear();
	xGrad->vVector.clear();
	y->vVector.clear();
	yGrad->vVector.clear();
	hTIn->vVector.clear();
	hTInGrad->vVector.clear();

}
#pragma endregion
#pragma region cnl
cnl::cnl() {

}
cnl::cnl(int _a, int _b) {

	this->a = _a;
	this->b = _b;

	ptr<model> nlr = neuronLR(0.3);
	this->filterTemplate = tnn({ a, b }, nlr);

	filterTemplate->x = make1D(a);
	filterTemplate->xGrad = make1D(a);
	filterTemplate->y = make1D(b);
	filterTemplate->yGrad = make1D(b);

	comp_LenA = make1D(a);

}
cnl::cnl(int _a, int _b, ptr<model> _filter) {
	
	this->a = _a;
	this->b = _b;

	this->filterTemplate = _filter;
	filterTemplate->x = make1D(a);
	filterTemplate->xGrad = make1D(a);
	filterTemplate->y = make1D(b);
	filterTemplate->yGrad = make1D(b);

	comp_LenA = make1D(a);

}
void cnl::fwd() {

	// pull in reference to vector to save compute
	vector<ptr<cType>>* xVec = &x->vVector;
	vector<ptr<cType>>* yVec = &y->vVector;

	int xVecSize = xVec->size();
	xGrad = make1D(xVecSize);
	assert(xVecSize >= a);
	y = make1D(ySize(xVecSize));

	for (int xIndex = 0; xIndex <= xVecSize - a; xIndex++) {
		
		model* m = (model*)at(xIndex).get();
		copy1D(x, m->x, xIndex, a, 0);
		m->fwd();
		copy1D(m->y, y, 0, b, xIndex * b);

	}

}
void cnl::bwd() {

	// pull in reference to vector to save compute
	vector<ptr<cType>>* xGradVec = &xGrad->vVector;
	vector<ptr<cType>>* yGradVec = &yGrad->vVector;

	int yGradVecSize = yGradVec->size();
	assert(yGradVecSize >= b);

	// clear the xGradVector
	clear1D(xGrad);

	for (int xGradIndex = 0; xGradIndex <= xGradVec->size() - a; xGradIndex++) {

		model* m = (model*)at(xGradIndex).get();
		copy(xGrad, comp_LenA, xGradIndex, a, 0);
		int yGradIndex = xGradIndex * b;
		copy1D(yGrad, m->yGrad, yGradIndex, b, 0);
		m->bwd();
		add1D(comp_LenA, m->xGrad, comp_LenA);

	}

}
ptr<model> cnl::clone() {

	cnl* result = new cnl(a, b, filterTemplate->clone());
	return result;

}
void cnl::modelWise(function<void(model*)> func) {

	func(this);
	filterTemplate->modelWise(func);

}
void cnl::prep(int a) {

	for (int i = 0; i < a; i++) {
		prepared.push_back(filterTemplate->clone());
	}

}
void cnl::unroll(int a) {

	for (int i = 0; i < a; i++) {
		push_back(prepared.at(size()));
	}

}
void cnl::clear() {
	vector<ptr<model>>::clear();
}
int cnl::numSteps(int xSize) {
	double d_x_size = (double)xSize;
	double d_a = (double)a;
	double d_num_strides = floor((d_x_size - d_a) + 1);
	if (d_num_strides < 0) {
		return 0;
	}
	else {
		return (int)d_num_strides;
	}
}
int cnl::ySize(int xSize) {

	double d_x_size = (double)xSize;
	double d_a = (double)a;
	double d_b = (double)b;
	double d_num_strides = floor((d_x_size - d_a) + 1);
	if (d_num_strides < 0) {
		return 0;
	}
	else {
		double d_y_size = d_b * d_num_strides;
		return (int)d_y_size;
	}

}
int cnl::xSize(int ySize) {

	double d_y_size = (double)ySize;
	double d_a = (double)a;
	double d_b = (double)b;
	double d_x_size = d_y_size / d_b + d_a;
	return (int)d_x_size;

}

#pragma endregion

#pragma endregion