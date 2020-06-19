#pragma once
#include "superHeader.h"
#include "optimization.h"
#include "maths.h"

class model;
class modelBpg;
class bias;
class biasBpg;
class act;
class actBpg;
class weight;
class weightBpg;
class wSet;
class wSetBpg;
class wJunc;
class wJuncBpg;
class seq;
class seqBpg;
class layer;
class layerBpg;

void initParam(model* m, vector<sPtr<sPtr<param>>>* paramVecOutput);

void modelFwd(sPtr<cType> x, sPtr<cType> y);
void modelBwd(sPtr<cType> yGrad, sPtr<cType> xGrad);

void biasFwd(sPtr<cType> x, sPtr<cType> y, sPtr<param> prm);
void biasBwd(sPtr<cType> yGrad, sPtr<cType> xGrad, sPtr<param> prm);

void actFwd(sPtr<cType> x, sPtr<cType> y, sPtr<actFunc> af);
void actBwd(sPtr<cType> yGrad, sPtr<cType> y, sPtr<cType> xGrad, sPtr<actFunc> af);

void weightFwd(sPtr<cType> x, sPtr<cType> y, sPtr<param> prm);
void weightBwd(sPtr<cType> yGrad, sPtr<cType> xGrad, sPtr<cType> x, sPtr<param> prm);

void wSetFwd(sPtr<cType> x, sPtr<cType> y, vector<sPtr<model>>* models);
void wSetBwd(sPtr<cType> yGrad, sPtr<cType> xGrad, vector<sPtr<model>>* models);

void wJuncFwd(sPtr<cType> x, sPtr<cType> y, vector<sPtr<model>>* models);
void wJuncBwd(sPtr<cType> yGrad, sPtr<cType> xGrad, vector<sPtr<model>>* models);

void seqFwd(sPtr<cType> x, sPtr<cType> y, vector<sPtr<model>>* models);
void seqBwd(sPtr<cType> yGrad, sPtr<cType> xGrad, vector<sPtr<model>>* models);

void layerFwd(sPtr<cType> x, sPtr<cType> y, vector<sPtr<model>>* models);
void layerBwd(sPtr<cType> yGrad, sPtr<cType> xGrad, vector<sPtr<model>>* models);

class model {
public:
	model();
	virtual void fwd();
	virtual void modelWise(function<void(model*)> func);
	virtual sPtr<model> clone();
	sPtr<cType> x;
	sPtr<cType> y;
};

class modelBpg : public model {
public:
	modelBpg();
	virtual void bwd();
	virtual sPtr<model> clone();
	sPtr<cType> xGrad;
	sPtr<cType> yGrad;
};

class bias : public model {
public:
	bias();
	virtual void fwd();
	virtual sPtr<model> clone();
	sPtr<sPtr<param>> prm;
};

class biasBpg : public modelBpg {
public:
	biasBpg();
	virtual void fwd();
	virtual void bwd();
	virtual sPtr<model> clone();
	sPtr<sPtr<param>> prm;
};

class act : public model {
public:
	act(actFunc* _af);
	act(sPtr<actFunc> _af);
	virtual void fwd();
	virtual sPtr<model> clone();
	sPtr<actFunc> af;
};

class actBpg : public modelBpg {
public:
	actBpg(actFunc* _af);
	actBpg(sPtr<actFunc> _af);
	virtual void fwd();
	virtual void bwd();
	virtual sPtr<model> clone();
	sPtr<actFunc> af;
};

class weight : public model {
public:
	weight();
	virtual void fwd();
	virtual sPtr<model> clone();
	sPtr<sPtr<param>> prm;
};

class weightBpg : public modelBpg {
public:
	weightBpg();
	virtual void fwd();
	virtual void bwd();
	virtual sPtr<model> clone();
	sPtr<sPtr<param>> prm;
};

class wSet : public model, public vector<sPtr<model>> {
public:
	wSet(int a);
	virtual void fwd();
	virtual void modelWise(function<void(model*)> func);
	virtual sPtr<model> clone();
};

class wSetBpg : public modelBpg, public vector<sPtr<model>> {
public:
	wSetBpg(int a);
	virtual void fwd();
	virtual void bwd();
	virtual void modelWise(function<void(model*)> func);
	virtual sPtr<model> clone();
};

class wJunc : public model, public vector<sPtr<model>> {
public:
	wJunc(int a, int b);
	virtual void fwd();
	virtual void modelWise(function<void(model*)> func);
	virtual sPtr<model> clone();
};

class wJuncBpg : public modelBpg, public vector<sPtr<model>> {
public:
	wJuncBpg(int a, int b);
	virtual void fwd();
	virtual void bwd();
	virtual void modelWise(function<void(model*)> func);
	virtual sPtr<model> clone();
};

class seq : public model, public vector<sPtr<model>> {
public:
	seq();
	virtual void fwd();
	virtual void modelWise(function<void(model*)> func);
	virtual sPtr<model> clone();
};

class seqBpg : public modelBpg, public vector<sPtr<model>> {
public:
	seqBpg();
	virtual void fwd();
	virtual void bwd();
	virtual void modelWise(function<void(model*)> func);
	virtual sPtr<model> clone();
};

class layer : public model, public vector<sPtr<model>> {
public:
	layer();
	layer(int a, model* modelTemplate);
	layer(int a, sPtr<model> modelTemplate);
	virtual void fwd();
	virtual void modelWise(function<void(model*)> func);
	virtual sPtr<model> clone();
};

class layerBpg : public modelBpg, public vector<sPtr<model>> {
public:
	layerBpg();
	layerBpg(int a, model* modelTemplate);
	layerBpg(int a, sPtr<model> modelTemplate);
	virtual void fwd();
	virtual void bwd();
	virtual void modelWise(function<void(model*)> func);
	virtual sPtr<model> clone();
};