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
class sync;
class syncBpg;
class lstmTS;
class lstmTSBpg;
class lstm;
class lstmBpg;
class muTS;
class muTSBpg;
class mu;
class muBpg;

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

void initParam(model* m, vector<ptr<ptr<param>>>* paramVecOutput);

class model {
public:
	model();
	virtual void fwd();
	virtual void modelWise(function<void(model*)> func);
	virtual ptr<model> clone();
	ptr<cType> x;
	ptr<cType> y;
};

class modelBpg : public model {
public:
	modelBpg();
	virtual void bwd();
	virtual ptr<model> clone();
	ptr<cType> xGrad;
	ptr<cType> yGrad;
};

class bias : public model {
public:
	bias();
	virtual void fwd();
	virtual ptr<model> clone();
	ptr<ptr<param>> prm;
};

class biasBpg : public modelBpg {
public:
	biasBpg();
	virtual void fwd();
	virtual void bwd();
	virtual ptr<model> clone();
	ptr<ptr<param>> prm;
};

class act : public model {
public:
	act();
	act(actFunc* _af);
	act(ptr<actFunc> _af);
	virtual void fwd();
	virtual ptr<model> clone();
	ptr<actFunc> af;
};

class actBpg : public modelBpg {
public:
	actBpg();
	actBpg(actFunc* _af);
	actBpg(ptr<actFunc> _af);
	virtual void fwd();
	virtual void bwd();
	virtual ptr<model> clone();
	ptr<actFunc> af;
};

class weight : public model {
public:
	weight();
	virtual void fwd();
	virtual ptr<model> clone();
	ptr<ptr<param>> prm;
};

class weightBpg : public modelBpg {
public:
	weightBpg();
	virtual void fwd();
	virtual void bwd();
	virtual ptr<model> clone();
	ptr<ptr<param>> prm;
};

class wSet : public model, public vector<ptr<model>> {
public:
	wSet();
	wSet(int _a);
	virtual void fwd();
	virtual void modelWise(function<void(model*)> func);
	virtual ptr<model> clone();
	int a;
};

class wSetBpg : public modelBpg, public vector<ptr<model>> {
public:
	wSetBpg();
	wSetBpg(int _a);
	virtual void fwd();
	virtual void bwd();
	virtual void modelWise(function<void(model*)> func);
	virtual ptr<model> clone();
	int a;
};

class wJunc : public model, public vector<ptr<model>> {
public:
	wJunc();
	wJunc(int _a, int _b);
	virtual void fwd();
	virtual void modelWise(function<void(model*)> func);
	virtual ptr<model> clone();
	int a;
	int b;
};

class wJuncBpg : public modelBpg, public vector<ptr<model>> {
public:
	wJuncBpg();
	wJuncBpg(int _a, int _b);
	virtual void fwd();
	virtual void bwd();
	virtual void modelWise(function<void(model*)> func);
	virtual ptr<model> clone();
	int a;
	int b;
};

class seq : public model, public vector<ptr<model>> {
public:
	seq();
	virtual void fwd();
	virtual void modelWise(function<void(model*)> func);
	virtual ptr<model> clone();
};

class seqBpg : public modelBpg, public vector<ptr<model>> {
public:
	seqBpg();
	virtual void fwd();
	virtual void bwd();
	virtual void modelWise(function<void(model*)> func);
	virtual ptr<model> clone();
};

class layer : public model, public vector<ptr<model>> {
public:
	layer();
	layer(int a, model* modelTemplate);
	layer(int a, ptr<model> modelTemplate);
	virtual void fwd();
	virtual void modelWise(function<void(model*)> func);
	virtual ptr<model> clone();
};

class layerBpg : public modelBpg, public vector<ptr<model>> {
public:
	layerBpg();
	layerBpg(int a, model* modelTemplate);
	layerBpg(int a, ptr<model> modelTemplate);
	virtual void fwd();
	virtual void bwd();
	virtual void modelWise(function<void(model*)> func);
	virtual ptr<model> clone();
};

class sync : public model, public vector<ptr<model>> {
public:
	sync();
	sync(model* _modelTemplate);
	virtual void fwd();
	virtual void modelWise(function<void(model*)> func);
	virtual ptr<model> clone();

	virtual void prep(int a);
	virtual void unroll(int a);

	// models that have been instantiated in RAM, and therefore are ready to be unrolled when ready to use
	vector<ptr<model>> prepared;
	// template model that will be cloned when prep() is called
	ptr<model> modelTemplate;
};

class syncBpg : public modelBpg, public vector<ptr<model>> {
public:
	syncBpg();
	syncBpg(model* _modelTemplate);
	virtual void fwd();
	virtual void bwd();
	virtual void modelWise(function<void(model*)> func);
	virtual ptr<model> clone();

	virtual void prep(int a);
	virtual void unroll(int a);

	// models that have been instantiated in RAM, and therefore are ready to be unrolled when ready to use
	vector<ptr<model>> prepared;
	// template model that will be cloned when prep() is called
	ptr<model> modelTemplate;
};

class lstmTS : public model {
public:
	lstmTS();
	lstmTS(int _units, ptr<model> _aGate, ptr<model> _bGate, ptr<model> _cGate, ptr<model> _dGate);
	virtual void fwd();
	virtual void modelWise(function<void(model*)> func);
	virtual ptr<model> clone();

	int units;

	ptr<model> aGate;
	ptr<model> bGate;
	ptr<model> cGate;
	ptr<model> dGate;

	ptr<cType> cTIn;
	ptr<cType> cTOut;
	ptr<cType> hTIn;
	ptr<cType> hTOut;
private:
	ptr<cType> comp_LenUnits;
	ptr<cType> comp_Len2Units;
};

class lstmTSBpg : public modelBpg {
public:
	lstmTSBpg();
	lstmTSBpg(int _units, ptr<model> _aGate, ptr<model> _bGate, ptr<model> _cGate, ptr<model> _dGate);
	virtual void fwd();
	virtual void bwd();
	virtual void modelWise(function<void(model*)> func);
	virtual ptr<model> clone();

	int units;

	ptr<model> aGate;
	ptr<model> bGate;
	ptr<model> cGate;
	ptr<model> dGate;

	ptr<cType> cTIn;
	ptr<cType> cTOut;
	ptr<cType> hTIn;
	ptr<cType> hTOut;

	ptr<cType> cTInGrad;
	ptr<cType> cTOutGrad;
	ptr<cType> hTInGrad;
	ptr<cType> hTOutGrad;
private:
	ptr<cType> comp_LenUnits;
	ptr<cType> comp_Len2Units;
};

class lstm : public model, public vector<ptr<model>> {
public:
	lstm();
	lstm(int _units);
	lstm(int _units, ptr<model> _aGate, ptr<model> _bGate, ptr<model> _cGate, ptr<model> _dGate);
	virtual void fwd();
	virtual void modelWise(function<void(model*)> func);
	virtual ptr<model> clone();

	virtual void prep(int a);
	virtual void unroll(int a);
	virtual void clear();

	int units;

	// models that have been instantiated in RAM, and therefore are ready to be unrolled when ready to use
	vector<ptr<model>> prepared;
	// template model that will be cloned when prep() is called
	ptr<model> lstmTSTemplate;

	ptr<cType> cTIn;
	ptr<cType> cTOut;
	ptr<cType> hTIn;
	ptr<cType> hTOut;
};

class lstmBpg : public modelBpg, public vector<ptr<model>> {
public:
	lstmBpg();
	lstmBpg(int _units);
	lstmBpg(int _units, ptr<model> _aGate, ptr<model> _bGate, ptr<model> _cGate, ptr<model> _dGate);
	virtual void fwd();
	virtual void bwd();
	virtual void modelWise(function<void(model*)> func);
	virtual ptr<model> clone();

	virtual void prep(int a);
	virtual void unroll(int a);
	virtual void clear();

	int units;

	// models that have been instantiated in RAM, and therefore are ready to be unrolled when ready to use
	vector<ptr<model>> prepared;
	// template model that will be cloned when prep() is called
	ptr<model> lstmTSTemplate;

	ptr<cType> cTIn;
	ptr<cType> cTOut;
	ptr<cType> hTIn;
	ptr<cType> hTOut;

	ptr<cType> cTInGrad;
	ptr<cType> cTOutGrad;
	ptr<cType> hTInGrad;
	ptr<cType> hTOutGrad;
};

class muTS : public model {
public:
	muTS();
	muTS(int _xUnits, int _cTUnits, int _hTUnits, ptr<model> _gate);
	virtual void fwd();
	virtual void modelWise(function<void(model*)> func);
	virtual ptr<model> clone();

	int xUnits;
	int cTUnits;
	int hTUnits;

	ptr<model> gate;

	ptr<cType> cTIn;
	ptr<cType> cTOut;
	ptr<cType> hTIn;
	ptr<cType> hTOut;
private:
	ptr<cType> comp_LenCTUnits;
};

class muTSBpg : public modelBpg {
public:
	muTSBpg();
	muTSBpg(int _xUnits, int _cTUnits, int _hTUnits, ptr<model> _gate);
	virtual void fwd();
	virtual void bwd();
	virtual void modelWise(function<void(model*)> func);
	virtual ptr<model> clone();

	int xUnits;
	int cTUnits;
	int hTUnits;

	ptr<model> gate;

	ptr<cType> cTIn;
	ptr<cType> cTOut;
	ptr<cType> hTIn;
	ptr<cType> hTOut;

	ptr<cType> cTInGrad;
	ptr<cType> cTOutGrad;
	ptr<cType> hTInGrad;
	ptr<cType> hTOutGrad;
private:
	ptr<cType> comp_LenCTUnits;
	ptr<cType> comp_LenHTUnits;
};

class mu : public model, public vector<ptr<model>> {
public:
	mu();
	mu(int _xUnits, int _cTUnits, int _hTUnits);
	mu(int _xUnits, int _cTUnits, int _hTUnits, ptr<model> _gate);
	virtual void fwd();
	virtual void modelWise(function<void(model*)> func);
	virtual ptr<model> clone();

	virtual void prep(int a);
	virtual void unroll(int a);
	virtual void clear();

	int xUnits;
	int cTUnits;
	int hTUnits;

	// models that have been instantiated in RAM, and therefore are ready to be unrolled when ready to use
	vector<ptr<model>> prepared;
	// template model that will be cloned when prep() is called
	ptr<model> muTSTemplate;

	ptr<cType> cTIn;
	ptr<cType> cTOut;
	ptr<cType> hTIn;
	ptr<cType> hTOut;
};

class muBpg : public modelBpg, public vector<ptr<model>> {
public:
	muBpg();
	muBpg(int _xUnits, int _cTUnits, int _hTUnits);
	muBpg(int _xUnits, int _cTUnits, int _hTUnits, ptr<model> gate);
	virtual void fwd();
	virtual void bwd();
	virtual void modelWise(function<void(model*)> func);
	virtual ptr<model> clone();

	virtual void prep(int a);
	virtual void unroll(int a);
	virtual void clear();

	int xUnits;
	int cTUnits;
	int hTUnits;

	// models that have been instantiated in RAM, and therefore are ready to be unrolled when ready to use
	vector<ptr<model>> prepared;
	// template model that will be cloned when prep() is called
	ptr<model> muTSTemplate;

	ptr<cType> cTIn;
	ptr<cType> cTOut;
	ptr<cType> hTIn;
	ptr<cType> hTOut;

	ptr<cType> cTInGrad;
	ptr<cType> cTOutGrad;
	ptr<cType> hTInGrad;
	ptr<cType> hTOutGrad;
};