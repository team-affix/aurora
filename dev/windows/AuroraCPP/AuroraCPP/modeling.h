#pragma once
#include "superHeader.h"
#include "optimization.h"
#include "maths.h"

void modelFwd(sPtr<cType> x, sPtr<cType> y);
void modelBwd(sPtr<cType> yGrad, sPtr<cType> xGrad);

void biasFwd(sPtr<cType> x, sPtr<cType> y, sPtr<param> prm);
void biasBwd(sPtr<cType> yGrad, sPtr<cType> xGrad, sPtr<param> prm);

void actFwd(sPtr<cType> x, sPtr<cType> y, sPtr<actFunc> af);
void actBwd(sPtr<cType> yGrad, sPtr<cType> y, sPtr<cType> xGrad, sPtr<actFunc> af);


class model;
class modelBpg;

class model {
public:
	model();
	virtual void fwd();
	virtual void modelWise(function<void(model*)> func);
	sPtr<cType> x;
	sPtr<cType> y;
};

class modelBpg : public model {
public:
	modelBpg();
	virtual void bwd();
	sPtr<cType> xGrad;
	sPtr<cType> yGrad;
};

class bias : public model {
public:
	bias();
	virtual void fwd();
	sPtr<param> prm;
};

class biasBpg : public modelBpg {
public:
	biasBpg();
	virtual void fwd();
	virtual void bwd();
	sPtr<param> prm;
};

class act : public model {
public:
	act(actFunc* _af);
	virtual void fwd();
	sPtr<actFunc> af;
};

class actBpg : public modelBpg {
public:
	actBpg(actFunc* _af);
	virtual void fwd();
	virtual void bwd();
	sPtr<actFunc> af;
};