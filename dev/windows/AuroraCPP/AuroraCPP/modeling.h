#pragma once
#include "superHeader.h"
#include "optimization.h"
#include "maths.h"

void modelFwd(sPtr<cType> x, sPtr<cType> y);
void modelBwd(sPtr<cType> yGrad, sPtr<cType> xGrad);

void biasFwd(sPtr<cType> x, sPtr<cType> y);
void biasBwd(sPtr<cType> yGrad, sPtr<cType> xGrad);



class model;
class modelBpg;

class model {
public:
	virtual void fwd();
	virtual void modelWise(function<void(model*)> func);
	sPtr<cType> x;
	sPtr<cType> y;
};

class modelBpg : public model {
public:
	virtual void bwd();
	sPtr<cType> xGrad;
	sPtr<cType> yGrad;
};

class bias : public model {
public:
	virtual void fwd();
};

class biasBpg : public modelBpg {
public:
	virtual void fwd();
	virtual void bwd();
};