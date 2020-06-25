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

seq* tnn(vector<int> npl, vector<model*> layerNeuronTemplates);
seqBpg* tnnBpg(vector<int> npl, vector<model*> layerNeuronTemplates);

seq* tnn(vector<int> npl, model* neuronTemplate);
seqBpg* tnnBpg(vector<int> npl, model* neuronTemplate);

seq neuronSm();
seq neuronTh();
seq neuronLR(double m);
seqBpg neuronSmBpg();
seqBpg neuronThBpg();
seqBpg neuronLRBpg(double m);

void initParam(model* m, vector<sPtr<sPtr<param>>>* paramVecOutput);

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
	act();
	act(actFunc* _af);
	act(sPtr<actFunc> _af);
	virtual void fwd();
	virtual sPtr<model> clone();
	sPtr<actFunc> af;
};

class actBpg : public modelBpg {
public:
	actBpg();
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
	wSet();
	wSet(int _a);
	virtual void fwd();
	virtual void modelWise(function<void(model*)> func);
	virtual sPtr<model> clone();
	int a;
};

class wSetBpg : public modelBpg, public vector<sPtr<model>> {
public:
	wSetBpg();
	wSetBpg(int _a);
	virtual void fwd();
	virtual void bwd();
	virtual void modelWise(function<void(model*)> func);
	virtual sPtr<model> clone();
	int a;
};

class wJunc : public model, public vector<sPtr<model>> {
public:
	wJunc();
	wJunc(int _a, int _b);
	virtual void fwd();
	virtual void modelWise(function<void(model*)> func);
	virtual sPtr<model> clone();
	int a;
	int b;
};

class wJuncBpg : public modelBpg, public vector<sPtr<model>> {
public:
	wJuncBpg();
	wJuncBpg(int _a, int _b);
	virtual void fwd();
	virtual void bwd();
	virtual void modelWise(function<void(model*)> func);
	virtual sPtr<model> clone();
	int a;
	int b;
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

class sync : public model, public vector<sPtr<model>> {
public:
	sync();
	sync(model* _modelTemplate);
	virtual void fwd();
	virtual void modelWise(function<void(model*)> func);
	virtual sPtr<model> clone();

	virtual void prep(int a);
	virtual void unroll(int a);

	// models that have been instantiated in RAM, and therefore are ready to be unrolled when ready to use
	vector<sPtr<model>> prepared;
	// template model that will be cloned when prep() is called
	sPtr<model> modelTemplate;
};

class syncBpg : public modelBpg, public vector<sPtr<model>> {
public:
	syncBpg();
	syncBpg(model* _modelTemplate);
	virtual void fwd();
	virtual void bwd();
	virtual void modelWise(function<void(model*)> func);
	virtual sPtr<model> clone();

	virtual void prep(int a);
	virtual void unroll(int a);

	// models that have been instantiated in RAM, and therefore are ready to be unrolled when ready to use
	vector<sPtr<model>> prepared;
	// template model that will be cloned when prep() is called
	sPtr<model> modelTemplate;
};

class lstmTS : public model {
public:
	lstmTS();
	lstmTS(int _units, sPtr<model> _aGate, sPtr<model> _bGate, sPtr<model> _cGate, sPtr<model> _dGate);
	virtual void fwd();
	virtual void modelWise(function<void(model*)> func);
	virtual sPtr<model> clone();

	int units;

	sPtr<model> aGate;
	sPtr<model> bGate;
	sPtr<model> cGate;
	sPtr<model> dGate;

	sPtr<cType> cTIn;
	sPtr<cType> cTOut;
	sPtr<cType> hTIn;
	sPtr<cType> hTOut;
};

class lstmTSBpg : public modelBpg {
public:
	lstmTSBpg();
	lstmTSBpg(int _units, sPtr<model> _aGate, sPtr<model> _bGate, sPtr<model> _cGate, sPtr<model> _dGate);
	virtual void fwd();
	virtual void bwd();
	virtual void modelWise(function<void(model*)> func);
	virtual sPtr<model> clone();

	int units;

	sPtr<model> aGate;
	sPtr<model> bGate;
	sPtr<model> cGate;
	sPtr<model> dGate;

	sPtr<cType> cTIn;
	sPtr<cType> cTOut;
	sPtr<cType> hTIn;
	sPtr<cType> hTOut;

	sPtr<cType> cTInGrad;
	sPtr<cType> cTOutGrad;
	sPtr<cType> hTInGrad;
	sPtr<cType> hTOutGrad;
};

class lstm : public model, public vector<sPtr<model>> {
public:
	lstm();
	lstm(int _units);
	lstm(int _units, sPtr<model> _aGate, sPtr<model> _bGate, sPtr<model> _cGate, sPtr<model> _dGate);
	virtual void fwd();
	virtual void modelWise(function<void(model*)> func);
	virtual sPtr<model> clone();

	virtual void prep(int a);
	virtual void unroll(int a);

	int units;

	// models that have been instantiated in RAM, and therefore are ready to be unrolled when ready to use
	vector<sPtr<model>> prepared;
	// template model that will be cloned when prep() is called
	sPtr<model> lstmTSTemplate;

	sPtr<cType> cTIn;
	sPtr<cType> cTOut;
	sPtr<cType> hTIn;
	sPtr<cType> hTOut;
};

class lstmBpg : public modelBpg, public vector<sPtr<model>> {
public:
	lstmBpg();
	lstmBpg(int _units);
	lstmBpg(int _units, sPtr<model> _aGate, sPtr<model> _bGate, sPtr<model> _cGate, sPtr<model> _dGate);
	virtual void fwd();
	virtual void bwd();
	virtual void modelWise(function<void(model*)> func);
	virtual sPtr<model> clone();

	virtual void prep(int a);
	virtual void unroll(int a);

	int units;

	// models that have been instantiated in RAM, and therefore are ready to be unrolled when ready to use
	vector<sPtr<model>> prepared;
	// template model that will be cloned when prep() is called
	sPtr<model> lstmTSTemplate;

	sPtr<cType> cTIn;
	sPtr<cType> cTOut;
	sPtr<cType> hTIn;
	sPtr<cType> hTOut;

	sPtr<cType> cTInGrad;
	sPtr<cType> cTOutGrad;
	sPtr<cType> hTInGrad;
	sPtr<cType> hTOutGrad;
};