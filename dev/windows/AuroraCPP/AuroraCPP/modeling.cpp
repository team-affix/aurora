#pragma once
#include "modeling.h"

#pragma region functions

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
modelBpg::modelBpg() {
	x = new cType(0);
	y = new cType(0);
	xGrad = new cType(0);
	yGrad = new cType(0);
}
void modelBpg::bwd() {
	modelBwd(yGrad, xGrad);
}
void model::modelWise(function<void(model*)> func) {
	func(this);
}
#pragma endregion
#pragma region bias
bias::bias() {
	x = new cType(0);
	y = new cType(0);
}
void bias::fwd() {
	biasFwd(x, y, prm);
}
biasBpg::biasBpg() {
	x = new cType(0);
	y = new cType(0);
	xGrad = new cType(0);
	yGrad = new cType(0);
}
void biasBpg::fwd() {
	biasFwd(x, y, prm);
}
void biasBpg::bwd() {
	biasBwd(yGrad, xGrad, prm);
}
#pragma endregion
#pragma region act
act::act(actFunc* _af) {
	x = new cType(0);
	y = new cType(0);
	this->af = _af;
}
void act::fwd() {
	actFwd(x, y, af);
}
actBpg::actBpg(actFunc* _af) {
	x = new cType(0);
	y = new cType(0);
	xGrad = new cType(0);
	yGrad = new cType(0);
	this->af = _af;
}
void actBpg::fwd() {
	actFwd(x, y, af);
}
void actBpg::bwd() {
	actBwd(yGrad, y, xGrad, af);
}
#pragma endregion

#pragma endregion