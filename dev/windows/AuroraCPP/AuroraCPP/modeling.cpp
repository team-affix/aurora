#pragma once
#include "modeling.h"

#pragma region functions

// functions outside of classes, so they are usable between those of similar types.

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
seqBpg* tnnBpg(vector<int> npl, vector<ptr<model>> layerNeuronTemplates) {
	seqBpg* result = new seqBpg();
	for (int i = 0; i < npl.size() - 1; i++) {
		result->push_back(new layerBpg(npl.at(i), layerNeuronTemplates.at(i)));
		result->push_back(new wJuncBpg(npl.at(i), npl.at(i + 1)));
	}
	result->push_back(new layerBpg(npl.back(), layerNeuronTemplates.back()));
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
seqBpg* tnnBpg(vector<int> npl, ptr<model> neuronTemplate) {
	seqBpg* result = new seqBpg();
	for (int i = 0; i < npl.size() - 1; i++) {
		result->push_back(new layerBpg(npl.at(i), neuronTemplate));
		result->push_back(new wJuncBpg(npl.at(i), npl.at(i + 1)));
	}
	result->push_back(new layerBpg(npl.back(), neuronTemplate));
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
seqBpg* neuronSmBpg() {

	// construct tanh neuron
	seqBpg* nsm = new seqBpg();
	nsm->push_back(new biasBpg());
	nsm->push_back(new actBpg(new actFuncSm()));
	return nsm;

}
seqBpg* neuronThBpg() {

	// construct tanh neuron
	seqBpg* nth = new seqBpg();
	nth->push_back(new biasBpg());
	nth->push_back(new actBpg(new actFuncTh()));
	return nth;

}
seqBpg* neuronLRBpg(double m) {

	// construct tanh neuron
	seqBpg* nlr = new seqBpg();
	nlr->push_back(new biasBpg());
	nlr->push_back(new actBpg(new actFuncLR(m)));
	return nlr;

}
void initParam(model* m, vector<ptr<ptr<param>>>* paramVecOutput) {
	if (bias* b = dynamic_cast<bias*>(m)) {
		b->prm = new ptr<param>();
		paramVecOutput->push_back(b->prm);
	}
	else if (biasBpg* b = dynamic_cast<biasBpg*>(m)) {
		b->prm = new ptr<param>();
		paramVecOutput->push_back(b->prm);
	}
	else if (weight* w = dynamic_cast<weight*>(m)) {
		w->prm = new ptr<param>();
		paramVecOutput->push_back(w->prm);
	}
	else if (weightBpg* w = dynamic_cast<weightBpg*>(m)) {
		w->prm = new ptr<param>();
		paramVecOutput->push_back(w->prm);
	}
}
#pragma endregion
#pragma region model
void modelFwd(ptr<cType> x, ptr<cType> y) {
	*y = *x;
}
void modelBwd(ptr<cType> yGrad, ptr<cType> xGrad) {
	*xGrad = *yGrad;
}
#pragma endregion
#pragma region bias
void biasFwd(ptr<cType> x, ptr<cType> y, ptr<param> prm) {
	y->vDouble = x->vDouble + prm->state;
}
void biasBwd(ptr<cType> yGrad, ptr<cType> xGrad, ptr<param> prm) {
	paramSgd* p = (paramSgd*)prm.get();
	p->gradient += yGrad->vDouble;
	xGrad->vDouble = yGrad->vDouble;
}
#pragma endregion
#pragma region act
void actFwd(ptr<cType> x, ptr<cType> y, ptr<actFunc> af) {
	y->vDouble = af->eval(x->vDouble);
}
void actBwd(ptr<cType> x, ptr<cType> yGrad, ptr<cType> y, ptr<cType> xGrad, ptr<actFunc> af) {
	xGrad->vDouble = yGrad->vDouble * af->deriv(&x->vDouble, &y->vDouble);
}
#pragma endregion
#pragma region weight
void weightFwd(ptr<cType> x, ptr<cType> y, ptr<param> prm) {
	y->vDouble = x->vDouble * prm->state;
}
void weightBwd(ptr<cType> yGrad, ptr<cType> xGrad, ptr<cType> x, ptr<param> prm) {
	paramSgd* p = (paramSgd*)prm.get();
	p->gradient += yGrad->vDouble * x->vDouble;
	xGrad->vDouble = yGrad->vDouble * prm->state;
}
#pragma endregion
#pragma region wSet
void wSetFwd(ptr<cType> x, ptr<cType> y, vector<ptr<model>>* models) {

	vector<ptr<cType>>* yVec = &y->vVector;
	for (int i = 0; i < models->size(); i++) {

		model* m = models->at(i).get();
		m->x->vDouble = x->vDouble;
		m->fwd();
		yVec->at(i)->vDouble = m->y->vDouble;

	}

}
void wSetBwd(ptr<cType> yGrad, ptr<cType> xGrad, vector<ptr<model>>* models) {

	// reset xGrad of weightSet to avoid overaccumulation
	xGrad->vDouble = 0;

	vector<ptr<cType>>* yGradVec = &yGrad->vVector;
	for (int i = 0; i < models->size(); i++) {

		modelBpg* m = (modelBpg*)models->at(i).get();
		m->yGrad->vDouble = yGradVec->at(i)->vDouble;
		m->bwd();
		xGrad->vDouble += m->xGrad->vDouble;

	}

}
void wSetModelWise(function<void(model*)> func, vector<ptr<model>>* models) {
	for (int i = 0; i < models->size(); i++) {
		models->at(i)->modelWise(func);
	}
}
#pragma endregion
#pragma region wJunc
void wJuncFwd(ptr<cType> x, ptr<cType> y, vector<ptr<model>>* models) {

	// reset all values in vector to zero, to ensure that no overaccumulation occurs
	clear1D(y);
	vector<ptr<cType>>* xVec = &x->vVector;
	for (int i = 0; i < models->size(); i++) {

		model* m = models->at(i).get();
		m->x->vDouble = xVec->at(i)->vDouble;
		m->fwd();

		// accumulates output
		add1D(y, m->y, y);

	}

}
void wJuncBwd(ptr<cType> yGrad, ptr<cType> xGrad, vector<ptr<model>>* models) {

	vector<ptr<cType>>* xGradVec = &xGrad->vVector;
	for (int i = 0; i < models->size(); i++) {

		modelBpg* m = (modelBpg*)models->at(i).get();
		m->yGrad->vVector = yGrad->vVector;
		m->bwd();
		xGradVec->at(i)->vDouble = m->xGrad->vDouble;

	}

}
void wJuncModelWise(function<void(model*)> func, vector<ptr<model>>* models) {
	for (int i = 0; i < models->size(); i++) {
		models->at(i)->modelWise(func);
	}
}
#pragma endregion
#pragma region seq
void seqFwd(ptr<cType> x, ptr<cType> y, vector<ptr<model>>* models) {

	ptr<cType> currentInput = x;
	for (int i = 0; i < models->size(); i++) {

		model* m = models->at(i).get();
		m->x = currentInput;
		m->fwd();
		currentInput = m->y;

	}
	*y = *currentInput;

}
void seqBwd(ptr<cType> yGrad, ptr<cType> xGrad, vector<ptr<model>>* models) {

	ptr<cType> currentGradient = yGrad;
	for (int i = models->size() - 1; i >= 0; i--) {

		modelBpg* m = (modelBpg*)models->at(i).get();
		m->yGrad = currentGradient;
		m->bwd();
		currentGradient = m->xGrad;

	}
	*xGrad = *currentGradient;

}
void seqModelWise(function<void(model*)> func, vector<ptr<model>>* models) {

	for (int i = 0; i < models->size(); i++) {

		models->at(i)->modelWise(func);

	}

}
#pragma endregion
#pragma region layer
void layerFwd(ptr<cType> x, ptr<cType> y, vector<ptr<model>>* models) {

	vector<ptr<cType>>* xVec = &x->vVector;
	vector<ptr<cType>>* yVec = &y->vVector;
	for (int i = 0; i < models->size(); i++) {

		model* m = models->at(i).get();
		m->x = xVec->at(i);
		m->fwd();
		yVec->at(i) = m->y;

	}

}
void layerBwd(ptr<cType> yGrad, ptr<cType> xGrad, vector<ptr<model>>* models) {

	vector<ptr<cType>>* xGradVec = &xGrad->vVector;
	vector<ptr<cType>>* yGradVec = &yGrad->vVector;
	for (int i = 0; i < models->size(); i++) {

		modelBpg* m = (modelBpg*)models->at(i).get();
		m->yGrad = yGradVec->at(i);
		m->bwd();
		xGradVec->at(i) = m->xGrad;

	}

}
void layerModelWise(function<void(model*)> func, vector<ptr<model>>* models) {
	for (int i = 0; i < models->size(); i++) {
		models->at(i)->modelWise(func);
	}
}
#pragma endregion
#pragma region sync
void syncIncFwd(ptr<cType> x, ptr<cType> y, vector<ptr<model>>* models, int* index, int a) {

	vector<ptr<cType>>* xVec = &x->vVector;
	vector<ptr<cType>>* yVec = &y->vVector;

	for (int j = 0; j < a; j++) {

		int i = *index;
		model* m = models->at(i).get();
		m->x = xVec->at(i);
		m->fwd();
		yVec->at(i) = m->y;
		(*index)++;

	}


}
void syncIncBwd(ptr<cType> yGrad, ptr<cType> xGrad, vector<ptr<model>>* models, int* index, int a) {

	vector<ptr<cType>>* xGradVec = &xGrad->vVector;
	vector<ptr<cType>>* yGradVec = &yGrad->vVector;

	for (int j = 0; j < a; j++) {

		(*index)--;
		int i = *index;
		modelBpg* m = (modelBpg*)models->at(i).get();
		m->yGrad = yGradVec->at(i);
		m->bwd();
		xGradVec->at(i) = m->xGrad;

	}

}
void syncFwd(ptr<cType> x, ptr<cType> y, vector<ptr<model>>* models, int* index) {

	syncIncFwd(x, y, models, index, models->size());

}
void syncBwd(ptr<cType> yGrad, ptr<cType> xGrad, vector<ptr<model>>* models, int* index) {

	syncIncBwd(yGrad, xGrad, models, index, *index);

}
void syncModelWise(function<void(model*)> func, ptr<model> modelTemplate) {
	modelTemplate->modelWise(func);
}
#pragma endregion
#pragma region lstmTS
void lstmTSFwd(ptr<cType> x, ptr<cType> cTIn, ptr<cType> hTIn,
	ptr<cType> comp_LenUnits, ptr<cType> comp_Len2Units,
	ptr<cType> y, ptr<cType> cTOut, ptr<cType> hTOut,
	ptr<model> aGate, ptr<model> bGate, ptr<model> cGate, ptr<model> dGate) {
	
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
void lstmTSBwd(int units,
	ptr<cType> comp_LenUnits, ptr<cType> comp_Len2Units,
	ptr<cType> cTOut, ptr<cType> cTIn,
	ptr<cType> yGrad, ptr<cType> cTOutGrad, ptr<cType> hTOutGrad,
	ptr<cType> xGrad, ptr<cType> cTInGrad, ptr<cType> hTInGrad,
	ptr<model> aGate, ptr<model> bGate, ptr<model> cGate, ptr<model> dGate) {

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

	// cast all gates to the modelBpgs that they are
	modelBpg* aGateBpg = (modelBpg*)aGate.get();
	modelBpg* bGateBpg = (modelBpg*)bGate.get();
	modelBpg* cGateBpg = (modelBpg*)cGate.get();
	modelBpg* dGateBpg = (modelBpg*)dGate.get();

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
void lstmTSModelWise(function<void(model*)> func, ptr<model> aGate, ptr<model> bGate, ptr<model> cGate, ptr<model> dGate) {
	aGate->modelWise(func);
	bGate->modelWise(func);
	cGate->modelWise(func);
	dGate->modelWise(func);
}
#pragma endregion
#pragma region lstm
void lstmModelWise(function<void(model*)> func, ptr<model> lstmTSTemplate) {

	lstmTSTemplate->modelWise(func);

}
#pragma endregion
#pragma region muTS
void muTSFwd(int xUnits, int cTUnits, int hTUnits, ptr<cType> comp_LenCTUnits, ptr<cType> x, ptr<cType> y, ptr<cType> cTIn, ptr<cType> cTOut, ptr<cType> hTIn, ptr<cType> hTOut, ptr<model> gate) {

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
void muTSBwd(int xUnits, int cTUnits, int hTUnits, ptr<cType> comp_LenCTUnits, ptr<cType> comp_LenHTUnits, ptr<cType> xGrad, ptr<cType> yGrad, ptr<cType> cTInGrad, ptr<cType> cTOutGrad, ptr<cType> hTInGrad, ptr<cType> hTOutGrad, ptr<model> gate) {

	// import compuation cTypes
	vector<ptr<cType>>* comp_LenCTUnitsVec = &comp_LenCTUnits->vVector;
	ptr<cType> cTCopyOperand = comp_LenCTUnitsVec->at(0);

	vector<ptr<cType>>* comp_LenHTUnitsVec = &comp_LenHTUnits->vVector;
	ptr<cType> hTAddOperand = comp_LenHTUnitsVec->at(0);

	modelBpg* gateBpg = (modelBpg*)gate.get();

	add1D(yGrad, hTOutGrad, hTAddOperand);
	copy1D(cTOutGrad, gateBpg->yGrad, 0, cTUnits, 0);
	copy1D(hTAddOperand, gateBpg->yGrad, 0, hTUnits, cTUnits);
	gateBpg->bwd();
	copy1D(gateBpg->xGrad, cTCopyOperand, xUnits, cTUnits, 0);
	add1D(cTCopyOperand, cTOutGrad, cTInGrad);
	copy1D(gateBpg->xGrad, xGrad, 0, xUnits, 0);
	copy1D(gateBpg->xGrad, hTInGrad, xUnits + cTUnits, hTUnits, 0);

}
void muTSModelWise(function<void(model*)> func, ptr<model> gate) {

	gate->modelWise(func);

}
#pragma endregion
#pragma region mu
void muModelWise(function<void(model*)> func, ptr<model> muTSTemplate) {

	muTSTemplate->modelWise(func);

}
#pragma endregion
#pragma region attTS
void attTSIncFwd(ptr<cType> x, ptr<cType> y, ptr<cType> hTIn, ptr<cType> comp_LenXUnits, int xUnits, int hTUnits, vector<ptr<model>>* unrolled, int* index, int a) {

	vector<ptr<cType>>* comp_LenXUnitsVec = &comp_LenXUnits->vVector;
	ptr<cType> x_yproduct = comp_LenXUnitsVec->at(0);

	// pull in references to input matrices to save compute
	vector<ptr<cType>>* xVec = &x->vVector;

	clear1D(y);

	for (int j = 0; j < a; j++) {

		int i = *index;
		model* m = (model*)unrolled->at(i).get();
		// set input to the model
		concat(xVec->at(i), hTIn, m->x);
		m->fwd();
		mult1D(m->y, xVec->at(i), x_yproduct);
		add1D(y, x_yproduct, y);

		(*index)++;

	}

}
void attTSIncBwd(ptr<cType> x, ptr<cType> yGrad, ptr<cType> xGrad, ptr<cType> hTInGrad, ptr<cType> comp_LenXUnits, ptr<cType> comp_LenHTUnits, int xUnits, int hTUnits, vector<ptr<model>>* unrolled, int* index, int a) {

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

		(*index)--;
		int i = *index;
		modelBpg* m = (modelBpg*)unrolled->at(i).get();

		mult1D(yGrad, xVec->at(i), m->yGrad);
		m->bwd();
		copy1D(m->xGrad, xGrad_copyResult, 0, xUnits, 0);
		copy1D(m->xGrad, hTInGrad_copyResult, xUnits, hTUnits, 0);
		add1D(hTInGrad, hTInGrad_copyResult, hTInGrad);
		mult1D(yGrad, m->y, y_yGradproduct);
		add1D(y_yGradproduct, xGrad_copyResult, xGradVec->at(i));

	}
}
void attTSFwd(ptr<cType> x, ptr<cType> y, ptr<cType> hTIn, ptr<cType> comp_LenXUnits, int xUnits, int hTUnits, vector<ptr<model>>* unrolled, int* index) {

	attTSIncFwd(x, y, hTIn, comp_LenXUnits, xUnits, hTUnits, unrolled, index, unrolled->size() - *index);

}
void attTSBwd(ptr<cType> x, ptr<cType> yGrad, ptr<cType> xGrad, ptr<cType> hTInGrad, ptr<cType> comp_LenXUnits, ptr<cType> comp_LenHTUnits, int xUnits, int hTUnits, vector<ptr<model>>* unrolled, int* index) {

	attTSIncBwd(x, yGrad, xGrad, hTInGrad, comp_LenXUnits, comp_LenHTUnits, xUnits, hTUnits, unrolled, index, *index);

}
void attTSModelWise(function<void(model*)> func, ptr<model> seqTemplate) {

	seqTemplate->modelWise(func);
	
}
#pragma endregion
#pragma region att
void attModelWise(function<void(model*)> func, ptr<model> attTSTemplate) {

	attTSTemplate->modelWise(func);

}
#pragma endregion

#pragma endregion

#pragma region definitions

// class function definitions

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
ptr<model> model::clone() {
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
ptr<model> modelBpg::clone() {
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
ptr<model> bias::clone() {
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
ptr<model> biasBpg::clone() {
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
act::act(ptr<actFunc> _af) {
	x = new cType(0);
	y = new cType(0);
	this->af = _af;
}
void act::fwd() {
	actFwd(x, y, af);
}
ptr<model> act::clone() {
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
actBpg::actBpg(ptr<actFunc> _af) {
	x = new cType(0);
	y = new cType(0);
	this->af = _af;
}
void actBpg::fwd() {
	actFwd(x, y, af);
}
void actBpg::bwd() {
	actBwd(x, yGrad, y, xGrad, af);
}
ptr<model> actBpg::clone() {
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
ptr<model> weight::clone() {
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
ptr<model> weightBpg::clone() {
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
ptr<model> wSet::clone() {
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
ptr<model> wSetBpg::clone() {
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
ptr<model> wJunc::clone() {
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
ptr<model> wJuncBpg::clone() {
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
ptr<model> seq::clone() {
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
ptr<model> seqBpg::clone() {
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
layer::layer(int a, ptr<model> modelTemplate) {

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
ptr<model> layer::clone() {
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
layerBpg::layerBpg(int a, ptr<model> modelTemplate) {

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
ptr<model> layerBpg::clone() {
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
	syncFwd(x, y, this, &index);
}
void sync::incFwd(int a) {

	syncIncFwd(x, y, this, &index, a);

}
void sync::modelWise(function<void(model*)> func) {
	func(this);
	syncModelWise(func, modelTemplate);
}
ptr<model> sync::clone() {
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
void sync::clear() {
	vector<ptr<model>>::clear();
	x->vVector.clear();
	y->vVector.clear();
	index = 0;
}

syncBpg::syncBpg() {
	prepared = vector<ptr<model>>();
}
syncBpg::syncBpg(model* _modelTemplate) {
	this->modelTemplate = _modelTemplate;
	prepared = vector<ptr<model>>();
}
syncBpg::syncBpg(ptr<model> _modelTemplate) {
	this->modelTemplate = _modelTemplate;
	prepared = vector<ptr<model>>();
}
void syncBpg::fwd() {
	syncFwd(x, y, this, &index);
}
void syncBpg::incFwd(int a) {

	syncIncFwd(x, y, this, &index, a);

}
void syncBpg::bwd() {
	syncBwd(yGrad, xGrad, this, &index);
}
void syncBpg::incBwd(int a) {

	syncIncBwd(yGrad, xGrad, this, &index, a);

}
void syncBpg::modelWise(function<void(model*)> func) {
	func(this);
	syncModelWise(func, modelTemplate);
}
ptr<model> syncBpg::clone() {
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
void syncBpg::clear() {
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

	this->aGate->x = make1D(2 * units);
	this->aGate->y = make1D(units);
	this->bGate->x = make1D(2 * units);
	this->bGate->y = make1D(units);
	this->cGate->x = make1D(2 * units);
	this->cGate->y = make1D(units);
	this->dGate->x = make1D(2 * units);
	this->dGate->y = make1D(units);

	this->cTIn = make1D(units);
	this->cTOut = make1D(units);
	this->hTIn = make1D(units);
	this->hTOut = make1D(units);

	comp_LenUnits = make2D(3, units);
	comp_Len2Units = make2D(1, 2 * units);

}
void lstmTS::fwd() {
	lstmTSFwd(x, cTIn, hTIn, comp_LenUnits, comp_Len2Units, y, cTOut, hTOut, aGate, bGate, cGate, dGate);
}
void lstmTS::modelWise(function<void(model*)> func) {
	func(this);
	lstmTSModelWise(func, aGate, bGate, cGate, dGate);
}
ptr<model> lstmTS::clone() {

	lstmTS* result = new lstmTS(units, aGate->clone(), bGate->clone(), cGate->clone(), dGate->clone());
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
lstmTSBpg::lstmTSBpg(int _units, ptr<model> _aGate, ptr<model> _bGate, ptr<model> _cGate, ptr<model> _dGate) {

	this->units = _units;
	this->aGate = _aGate;
	this->bGate = _bGate;
	this->cGate = _cGate;
	this->dGate = _dGate;

	this->x = make1D(units);
	this->y = make1D(units);
	this->xGrad = make1D(units);
	this->yGrad = make1D(units);

	// cast all gates to the modelBpgs that they are
	modelBpg* aGateBpg = (modelBpg*)aGate.get();
	modelBpg* bGateBpg = (modelBpg*)bGate.get();
	modelBpg* cGateBpg = (modelBpg*)cGate.get();
	modelBpg* dGateBpg = (modelBpg*)dGate.get();

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
void lstmTSBpg::fwd() {
	lstmTSFwd(x, cTIn, hTIn, comp_LenUnits, comp_Len2Units, y, cTOut, hTOut, aGate, bGate, cGate, dGate);
}
void lstmTSBpg::bwd() {
	lstmTSBwd(units, comp_LenUnits, comp_Len2Units,
		cTOut, cTIn, yGrad,
		cTOutGrad, hTOutGrad, xGrad, cTInGrad, hTInGrad, 
		aGate, bGate, cGate, dGate);
}
void lstmTSBpg::modelWise(function<void(model*)> func) {
	func(this);
	lstmTSModelWise(func, aGate, bGate, cGate, dGate);
}
ptr<model> lstmTSBpg::clone() {

	lstmTSBpg* result = new lstmTSBpg(units, aGate->clone(), bGate->clone(), cGate->clone(), dGate->clone());
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

	this->lstmTSTemplate = new lstmTS(units, aGate, bGate, cGate, dGate);

}
//lstm::lstm(int _units, vector<int> _gateHiddenLayers) {
//
//	this->units = _units;
//
//	ptr<model> nlr = neuronLR(0.3);
//	ptr<model> nth = neuronTh();
//	ptr<model> nsm = neuronSm();
//
//	vector<int> gateDims = concat(concat({ 2 * units }, _gateHiddenLayers), { units });
//
//	ptr<model> aGate = tnn(gateDims, { nlr, nsm });
//	ptr<model> bGate = tnn(gateDims, { nlr, nsm });
//	ptr<model> cGate = tnn(gateDims, { nlr, nth });
//	ptr<model> dGate = tnn(gateDims, { nlr, nsm });
//
//	this->hTIn = make1D(units);
//	this->cTIn = make1D(units);
//	this->hTOut = make1D(units);
//	this->cTOut = make1D(units);
//
//	this->lstmTSTemplate = new lstmTS(units, aGate, bGate, cGate, dGate);
//
//}
//lstm::lstm(int _units, vector<int> _aGateHiddenLayers, vector<int> _bGateHiddenLayers, vector<int> _cGateHiddenLayers, vector<int> _dGateHiddenLayers) {
//
//}
lstm::lstm(int _units, ptr<model> _aGate, ptr<model> _bGate, ptr<model> _cGate, ptr<model> _dGate) {
	
	this->units = _units;

	this->hTIn = make1D(units);
	this->cTIn = make1D(units);
	this->hTOut = make1D(units);
	this->cTOut = make1D(units);

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
void lstm::modelWise(function<void(model*)> func) {

	func(this);
	lstmModelWise(func, lstmTSTemplate);

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
	}
}
void lstm::clear() {
	vector<ptr<model>>::clear();
	x->vVector.clear();
	y->vVector.clear();
	clear1D(cTIn);
	clear1D(hTIn);
	clear1D(cTOut);
	clear1D(hTOut);
	index = 0;
}

lstmBpg::lstmBpg() {

}
lstmBpg::lstmBpg(int _units) {

	this->units = _units;

	ptr<model> nlr = neuronLRBpg(0.3);
	ptr<model> nth = neuronThBpg();
	ptr<model> nsm = neuronSmBpg();

	ptr<model> aGate = tnnBpg({ 2 * units, units }, { nlr, nsm });
	ptr<model> bGate = tnnBpg({ 2 * units, units }, { nlr, nsm });
	ptr<model> cGate = tnnBpg({ 2 * units, units }, { nlr, nth });
	ptr<model> dGate = tnnBpg({ 2 * units, units }, { nlr, nsm });

	this->hTIn = make1D(units);
	this->cTIn = make1D(units);
	this->hTOut = make1D(units);
	this->cTOut = make1D(units);

	this->hTInGrad = make1D(units);
	this->cTInGrad = make1D(units);
	this->hTOutGrad = make1D(units);
	this->cTOutGrad = make1D(units);

	this->lstmTSTemplate = new lstmTSBpg(units, aGate, bGate, cGate, dGate);

}
lstmBpg::lstmBpg(int _units, ptr<model> _aGate, ptr<model> _bGate, ptr<model> _cGate, ptr<model> _dGate) {

	this->units = _units;

	this->hTIn = make1D(units);
	this->cTIn = make1D(units);
	this->hTOut = make1D(units);
	this->cTOut = make1D(units);

	this->hTInGrad = make1D(units);
	this->cTInGrad = make1D(units);
	this->hTOutGrad = make1D(units);
	this->cTOutGrad = make1D(units);

	this->lstmTSTemplate = new lstmTSBpg(units, _aGate, _bGate, _cGate, _dGate);

}
void lstmBpg::fwd() {

	incFwd(size() - index);

}
void lstmBpg::incFwd(int a) {

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

		lstmTSBpg* l = (lstmTSBpg*)at(index).get();
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
void lstmBpg::bwd() {

	incBwd(index);

}
void lstmBpg::incBwd(int a) {

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

		lstmTSBpg* l = (lstmTSBpg*)at(index).get();
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
void lstmBpg::modelWise(function<void(model*)> func) {
	
	func(this);
	lstmModelWise(func, lstmTSTemplate);

}
ptr<model> lstmBpg::clone() {

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
void lstmBpg::clear() {
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

	this->gate->x = make1D(xUnits + cTUnits + hTUnits);
	this->gate->y = make1D(cTUnits + hTUnits);

	this->cTIn = make1D(cTUnits);
	this->cTOut = make1D(cTUnits);
	this->hTIn = make1D(hTUnits);
	this->hTOut = make1D(hTUnits);

	this->comp_LenCTUnits = make2D(1, cTUnits);

}
void muTS::fwd() {

	muTSFwd(xUnits, cTUnits, hTUnits, comp_LenCTUnits, x, y, cTIn, cTOut, hTIn, hTOut, gate);

}
void muTS::modelWise(function<void(model*)> func) {

	func(this);
	muTSModelWise(func, gate);

}
ptr<model> muTS::clone() {

	muTS* result = new muTS(xUnits, cTUnits, hTUnits, gate->clone());
	return result;

}

muTSBpg::muTSBpg() {

}
muTSBpg::muTSBpg(int _xUnits, int _cTUnits, int _hTUnits, ptr<model> _gate) {

	this->xUnits = _xUnits;
	this->cTUnits = _cTUnits;
	this->hTUnits = _hTUnits;
	this->gate = _gate;

	this->x = make1D(xUnits);
	this->y = make1D(hTUnits);
	this->xGrad = make1D(xUnits);
	this->yGrad = make1D(hTUnits);

	modelBpg* gateBpg = (modelBpg*)gate.get();

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
void muTSBpg::fwd() {

	muTSFwd(xUnits, cTUnits, hTUnits, comp_LenCTUnits, x, y, cTIn, cTOut, hTIn, hTOut, gate);

}
void muTSBpg::bwd() {

	muTSBwd(xUnits, cTUnits, hTUnits, comp_LenCTUnits, comp_LenHTUnits, xGrad, yGrad, cTInGrad, cTOutGrad, hTInGrad, hTOutGrad, gate);

}
void muTSBpg::modelWise(function<void(model*)> func) {

	func(this);
	muTSModelWise(func, gate);

}
ptr<model> muTSBpg::clone() {

	muTSBpg* result = new muTSBpg(xUnits, cTUnits, hTUnits, gate->clone());
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

	ptr<model> nlr = neuronLR(0.3);

	ptr<model> gateTemplate = tnn({ xUnits + cTUnits + hTUnits, cTUnits + hTUnits }, { nlr, nlr });
	muTSTemplate = new muTS(xUnits, cTUnits, hTUnits, gateTemplate);

}
mu::mu(int _xUnits, int _cTUnits, int _hTUnits, ptr<model> _gate) {

	this->xUnits = _xUnits;
	this->cTUnits = _cTUnits;
	this->hTUnits = _hTUnits;

	this->cTIn = make1D(cTUnits);
	this->cTOut = make1D(cTUnits);
	this->hTIn = make1D(hTUnits);
	this->hTOut = make1D(hTUnits);

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
void mu::modelWise(function<void(model*)> func) {

	func(this);
	muModelWise(func, muTSTemplate);

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
	index = 0;
}

muBpg::muBpg() {

}
muBpg::muBpg(int _xUnits, int _cTUnits, int _hTUnits) {

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

	ptr<model> nlr = neuronLRBpg(0.3);

	ptr<model> gateTemplate = tnnBpg({ xUnits + cTUnits + hTUnits, cTUnits + hTUnits }, { nlr, nlr });
	muTSTemplate = new muTSBpg(xUnits, cTUnits, hTUnits, gateTemplate);

}
muBpg::muBpg(int _xUnits, int _cTUnits, int _hTUnits, ptr<model> _gate) {

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

	muTSTemplate = new muTSBpg(xUnits, cTUnits, hTUnits, _gate);

}
void muBpg::fwd() {

	incFwd(size() - index);

}
void muBpg::incFwd(int a) {

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

		muTSBpg* m = (muTSBpg*)at(index).get();
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
void muBpg::bwd() {

	incBwd(index);

}
void muBpg::incBwd(int a) {

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

		muTSBpg* m = (muTSBpg*)at(index).get();
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
void muBpg::modelWise(function<void(model*)> func) {

	func(this);
	muModelWise(func, muTSTemplate);

}
ptr<model> muBpg::clone() {

	muTSBpg* castedTemplate = (muTSBpg*)muTSTemplate.get();
	muBpg* result = new muBpg(xUnits, cTUnits, hTUnits, castedTemplate->gate->clone());
	return result;

}
void muBpg::prep(int a) {
	for (int i = 0; i < a; i++) {
		prepared.push_back(muTSTemplate->clone());
	}
}
void muBpg::unroll(int a) {
	for (int i = 0; i < a; i++) {
		push_back(prepared.at(size()));
		x->vVector.push_back(new cType{});
		y->vVector.push_back(new cType{});
		xGrad->vVector.push_back(new cType{});
		yGrad->vVector.push_back(new cType{});
	}
}
void muBpg::clear() {
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

	// Initialize y, and hT vectors
	this->hTIn = make1D(hTUnits);
	this->y = make1D(xUnits);

	// Initialize the prepared vector
	this->prepared = vector<ptr<model>>();

	// Instantiate neurons that will be used in attention template TNN
	ptr<model> nlr = neuronLR(0.3);
	ptr<model> nsm = neuronSm();

	// Initialize the attention timestep template model to be a tnn
	seq* seqTemp = tnn({ xUnits + hTUnits, xUnits }, { nlr, nsm });
	seqTemp->x = make1D(xUnits + hTUnits);
	seqTemp->y = make1D(xUnits);
	this->seqTemplate = seqTemp;

	// Initialize the computation result cType
	comp_LenXUnits = make2D(1, xUnits);

}
attTS::attTS(int _xUnits, int _hTUnits, ptr<model> _seqTemplate) {

	// Make the xUnits and hTUnits publicly accessable
	this->xUnits = _xUnits;
	this->hTUnits = _hTUnits;

	// Initialize x, y, and hT vectors
	this->hTIn = make1D(hTUnits);
	this->y = make1D(xUnits);

	// Initialize the prepared vector
	this->prepared = vector<ptr<model>>();

	// Initialize the attention timestep template model to be a tnn
	seq* seqTemp = (seq*)_seqTemplate.get();
	seqTemp->x = make1D(xUnits + hTUnits);
	seqTemp->y = make1D(xUnits);
	this->seqTemplate = seqTemp;

	// Initialize the computation result cType
	comp_LenXUnits = make2D(1, xUnits);

}
void attTS::fwd() {

	attTSFwd(x, y, hTIn, comp_LenXUnits, xUnits, hTUnits, this, &index);

}
void attTS::incFwd(int a) {

	attTSIncFwd(x, y, hTIn, comp_LenXUnits, xUnits, hTUnits, this, &index, a);

}
void attTS::modelWise(function<void(model*)> func) {
	
	func(this);
	attTSModelWise(func, seqTemplate);

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
	}

}
void attTS::clear() {

	vector<ptr<model>>::clear();
	x->vVector.clear();
	index = 0;

}

attTSBpg::attTSBpg() {

}
attTSBpg::attTSBpg(int _xUnits, int _hTUnits) {

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
	ptr<model> nlr = neuronLRBpg(0.3);
	ptr<model> nsm = neuronSmBpg();

	// Initialize the attention timestep template model to be a tnn
	seqBpg* seqTemp = tnnBpg({ xUnits + hTUnits, xUnits }, { nlr, nsm });
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
attTSBpg::attTSBpg(int _xUnits, int _hTUnits, ptr<model> _seqTemplate) {

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
	seqBpg* seqTemp = (seqBpg*)_seqTemplate.get();
	seqTemp->x = make1D(xUnits + hTUnits);
	seqTemp->y = make1D(xUnits);
	seqTemp->xGrad = make1D(xUnits + hTUnits);
	seqTemp->yGrad = make1D(xUnits);
	this->seqTemplate = seqTemp;

	// Initialize the computation result cType
	comp_LenXUnits = make2D(2, xUnits);
	comp_LenHTUnits = make2D(1, hTUnits);

}
void attTSBpg::fwd() {

	attTSFwd(x, y, hTIn, comp_LenXUnits, xUnits, hTUnits, this, &index);

}
void attTSBpg::incFwd(int a) {

	attTSIncFwd(x, y, hTIn, comp_LenXUnits, xUnits, hTUnits, this, &index, a);

}
void attTSBpg::bwd() {

	attTSBwd(x, yGrad, xGrad, hTInGrad, comp_LenXUnits, comp_LenHTUnits, xUnits, hTUnits, this, &index);

}
void attTSBpg::incBwd(int a) {

	attTSIncBwd(x, yGrad, xGrad, hTInGrad, comp_LenXUnits, comp_LenHTUnits, xUnits, hTUnits, this, &index, a);

}
void attTSBpg::modelWise(function<void(model*)> func) {

	func(this);
	attTSModelWise(func, seqTemplate);

}
ptr<model> attTSBpg::clone() {

	attTSBpg* result = new attTSBpg(xUnits, hTUnits, seqTemplate);
	return result;

}
void attTSBpg::prep(int a) {

	for (int i = 0; i < a; i++) {
		prepared.push_back(seqTemplate->clone());
	}

}
void attTSBpg::unroll(int a) {

	for (int i = 0; i < a; i++) {
		push_back(prepared.at(size()));
		x->vVector.push_back(make1D(xUnits));
		xGrad->vVector.push_back(make1D(xUnits));
	}

}
void attTSBpg::clear() {

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

	// Initialize the prepared vector
	this->prepared = vector<ptr<model>>();

	// Initialize template model
	this->attTSTemplate = new attTS(xUnits, hTUnits);
	attTSTemplate->y = make1D(xUnits);

}
att::att(int _xUnits, int _hTUnits, ptr<model> _attTSTemplate) {

	// Make the xUnits and hTUnits publicly accessable
	this->xUnits = _xUnits;
	this->hTUnits = _hTUnits;

	this->x = new cType{};
	this->hTIn = new cType{};

	// Initialize the prepared vector
	this->prepared = vector<ptr<model>>();

	// Initialize template model
	this->attTSTemplate = _attTSTemplate;
	attTSTemplate->y = make1D(xUnits);

}
void att::fwd() {

	incFwd(size() - index);

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
		y->vVector.push_back(new cType{});
		hTIn->vVector.push_back(new cType{});
		// x's vVector is not pushed back to because it is completely populated by the user before any timestep is carried forward

	}

}
void att::unroll(int a, int b) {

	for (int i = 0; i < a; i++) {

		ptr<model> m = prepared.at(size());
		attTS* att = (attTS*)m.get();
		att->unroll(b);
		push_back(m);
		x->vVector.push_back(make1D(xUnits));
		y->vVector.push_back(new cType{});
		hTIn->vVector.push_back(new cType{});
		// x's vVector is not pushed back to because it is completely populated by the user before any timestep is carried forward

	}

}
void att::clear() {

	vector<ptr<model>>::clear();
	x->vVector.clear();
	y->vVector.clear();
	hTIn->vVector.clear();

}

attBpg::attBpg() {

}
attBpg::attBpg(int _xUnits, int _hTUnits) {

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
	attTSBpg* attTSTemplateBpg = new attTSBpg(xUnits, hTUnits);
	attTSTemplateBpg->y = make1D(xUnits);
	attTSTemplateBpg->yGrad = make1D(xUnits);
	this->attTSTemplate = attTSTemplateBpg;

}
attBpg::attBpg(int _xUnits, int _hTUnits, ptr<model> _attTSTemplate) {

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
	attTSBpg* attTSTemplateBpg = (attTSBpg*)_attTSTemplate.get();
	attTSTemplateBpg->y = make1D(xUnits);
	attTSTemplateBpg->yGrad = make1D(xUnits);
	this->attTSTemplate = attTSTemplateBpg;

}
void attBpg::fwd() {

	// pull in reference to vector to save compute
	vector<ptr<cType>>* yVec = &y->vVector;
	// In reality, this vector is a vector of vectors, so a matrix.
	vector<ptr<cType>>* hTInMat = &hTIn->vVector;

	attTSBpg* a = (attTSBpg*)at(index).get();
	a->x = x;
	a->hTIn = hTInMat->at(index);
	a->fwd();
	yVec->at(index) = a->y;

	index++;

}
void attBpg::incFwd(int _a) {

	// pull in reference to vector to save compute
	vector<ptr<cType>>* yVec = &y->vVector;
	// In reality, this vector is a vector of vectors, so a matrix.
	vector<ptr<cType>>* hTInMat = &hTIn->vVector;

	for (int j = 0; j < _a; j++) {

		attTSBpg* a = (attTSBpg*)at(index).get();
		a->x = x;
		a->hTIn = hTInMat->at(index);
		a->fwd();
		yVec->at(index) = a->y;

		index++;

	}

}
void attBpg::bwd() {

	incBwd(index);

}
void attBpg::incBwd(int _a) {

	// pull in reference to vector to save compute
	vector<ptr<cType>>* yGradVec = &yGrad->vVector;
	vector<ptr<cType>>* xGradVec = &xGrad->vVector;
	// In reality, this vector is a vector of vectors, so a matrix.
	vector<ptr<cType>>* hTInGradMat = &hTInGrad->vVector;

	for (int j = 0; j < _a; j++) {

		index--;

		attTSBpg* a = (attTSBpg*)at(index).get();
		a->yGrad = yGradVec->at(index);
		a->bwd();
		add2D(xGrad, a->xGrad, xGrad);
		hTInGradMat->at(index) = a->hTInGrad;

	}

}
void attBpg::modelWise(function<void(model*)> func) {

	func(this);
	attTSTemplate->modelWise(func);

}
ptr<model> attBpg::clone() {

	attBpg* result = new attBpg(xUnits, hTUnits, attTSTemplate->clone());
	return result;

}
void attBpg::prep(int a) {

	for (int i = 0; i < a; i++) {
		prepared.push_back(attTSTemplate->clone());
	}

}
void attBpg::prep(int a, int b) {

	for (int i = 0; i < a; i++) {
		ptr<model> m = attTSTemplate->clone();
		attTSBpg* att = (attTSBpg*)m.get();
		att->prep(b);
		prepared.push_back(m);
	}

}
void attBpg::unroll(int a) {

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
void attBpg::unroll(int a, int b) {

	for (int i = 0; i < a; i++) {

		int s = size();
		ptr<model> m = prepared.at(size());
		attTSBpg* att = (attTSBpg*)m.get();
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
void attBpg::clear() {

	vector<ptr<model>>::clear();
	x->vVector.clear();
	xGrad->vVector.clear();
	y->vVector.clear();
	yGrad->vVector.clear();
	hTIn->vVector.clear();
	hTInGrad->vVector.clear();

}
#pragma endregion

#pragma endregion
