#pragma once
#include "superHeader.h"
#include "optimization.h"
#include "maths.h"

#pragma region Defs

#define MODELFIELDS \
virtual void fwd(); \
virtual ptr<model> clone();

#define MODELBPGFIELDS MODELFIELDS \
virtual void bwd();

#define SEQFIELDS MODELFIELDS \
virtual void modelWise(function<void(model*)> func);

#define SEQBPGFIELDS MODELBPGFIELDS \
virtual void modelWise(function<void(model*)> func);

#define RECFIELDS SEQFIELDS \
virtual void incFwd(int a);\
virtual void prep(int a);\
virtual void unroll(int a);\
virtual void clear(); \
int index;

#define RECBPGFIELDS SEQBPGFIELDS \
virtual void incFwd(int a);\
virtual void incBwd(int a);\
virtual void prep(int a);\
virtual void unroll(int a);\
virtual void clear(); \
int index;

#pragma endregion

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
class attTS;
class attTSBpg;

seq* tnn(vector<int> npl, vector<ptr<model>> layerNeuronTemplates);
seqBpg* tnnBpg(vector<int> npl, vector<ptr<model>> layerNeuronTemplates);

seq* tnn(vector<int> npl, ptr<model> neuronTemplate);
seqBpg* tnnBpg(vector<int> npl, ptr<model> neuronTemplate);

seq* neuronSm();
seq* neuronTh();
seq* neuronLR(double m);
seqBpg* neuronSmBpg();
seqBpg* neuronThBpg();
seqBpg* neuronLRBpg(double m);

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
	MODELFIELDS
	bias();
	ptr<ptr<param>> prm;
};
class biasBpg : public modelBpg {
public:
	MODELBPGFIELDS
	biasBpg();
	ptr<ptr<param>> prm;
};
class act : public model {
public:
	MODELFIELDS
	act();
	act(actFunc* _af);
	act(ptr<actFunc> _af);
	ptr<actFunc> af;
};
class actBpg : public modelBpg {
public:
	MODELBPGFIELDS
	actBpg();
	actBpg(actFunc* _af);
	actBpg(ptr<actFunc> _af);
	ptr<actFunc> af;
};
class weight : public model {
public:
	MODELFIELDS
	weight();
	ptr<ptr<param>> prm;
};
class weightBpg : public modelBpg {
public:
	MODELBPGFIELDS
	weightBpg();
	ptr<ptr<param>> prm;
};
class wSet : public model, public vector<ptr<model>> {
public:
	SEQFIELDS
	wSet();
	wSet(int _a);
	int a;
};
class wSetBpg : public modelBpg, public vector<ptr<model>> {
public:
	SEQBPGFIELDS
	wSetBpg();
	wSetBpg(int _a);
	int a;
};
class wJunc : public model, public vector<ptr<model>> {
public:
	SEQFIELDS
	wJunc();
	wJunc(int _a, int _b);
	int a;
	int b;
};
class wJuncBpg : public modelBpg, public vector<ptr<model>> {
public:
	SEQBPGFIELDS
	wJuncBpg();
	wJuncBpg(int _a, int _b);
	int a;
	int b;
};
class seq : public model, public vector<ptr<model>> {
public:
	SEQFIELDS
	seq();
};
class seqBpg : public modelBpg, public vector<ptr<model>> {
public:
	SEQBPGFIELDS
	seqBpg();
};
class layer : public model, public vector<ptr<model>> {
public:
	SEQFIELDS
	layer();
	layer(int a, model* modelTemplate);
	layer(int a, ptr<model> modelTemplate);
};
class layerBpg : public modelBpg, public vector<ptr<model>> {
public:
	SEQBPGFIELDS
	layerBpg();
	layerBpg(int a, model* modelTemplate);
	layerBpg(int a, ptr<model> modelTemplate);
};
class sync : public model, public vector<ptr<model>> {
public:
	RECFIELDS
	sync();
	sync(model* _modelTemplate);

	// models that have been instantiated in RAM, and therefore are ready to be unrolled when ready to use
	vector<ptr<model>> prepared;
	// template model that will be cloned when prep() is called
	ptr<model> modelTemplate;
};
class syncBpg : public modelBpg, public vector<ptr<model>> {
public:
	RECBPGFIELDS
	syncBpg();
	syncBpg(model* _modelTemplate);

	// models that have been instantiated in RAM, and therefore are ready to be unrolled when ready to use
	vector<ptr<model>> prepared;
	// template model that will be cloned when prep() is called
	ptr<model> modelTemplate;
};
class lstmTS : public model {
public:
	SEQFIELDS
	lstmTS();
	lstmTS(int _units, ptr<model> _aGate, ptr<model> _bGate, ptr<model> _cGate, ptr<model> _dGate);

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
	SEQBPGFIELDS
	lstmTSBpg();
	lstmTSBpg(int _units, ptr<model> _aGate, ptr<model> _bGate, ptr<model> _cGate, ptr<model> _dGate);

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
	RECFIELDS
	lstm();
	lstm(int _units);
	lstm(int _units, ptr<model> _aGate, ptr<model> _bGate, ptr<model> _cGate, ptr<model> _dGate);

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
	RECBPGFIELDS
	lstmBpg();
	lstmBpg(int _units);
	lstmBpg(int _units, ptr<model> _aGate, ptr<model> _bGate, ptr<model> _cGate, ptr<model> _dGate);

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
	SEQFIELDS
	muTS();
	muTS(int _xUnits, int _cTUnits, int _hTUnits, ptr<model> _gate);

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
	SEQBPGFIELDS
	muTSBpg();
	muTSBpg(int _xUnits, int _cTUnits, int _hTUnits, ptr<model> _gate);

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
	RECFIELDS
	mu();
	mu(int _xUnits, int _cTUnits, int _hTUnits);
	mu(int _xUnits, int _cTUnits, int _hTUnits, ptr<model> _gate);

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
	RECBPGFIELDS
	muBpg();
	muBpg(int _xUnits, int _cTUnits, int _hTUnits);
	muBpg(int _xUnits, int _cTUnits, int _hTUnits, ptr<model> gate);

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
class attTS : public model, public vector<ptr<model>> {
public:
	RECFIELDS
	attTS();
	attTS(int _xUnits, int _hTUnits);
	attTS(int _xUnits, int _hTUnits, ptr<model> _seqTemplate);

	int xUnits;
	int hTUnits;
	// models that have been instantiated in RAM, and therefore are ready to be unrolled when ready to use
	vector<ptr<model>> prepared;
	// template model that will be cloned when prep() is called
	ptr<model> seqTemplate;
	// the input hidden state is a vector, because it intakes one vector representing the decoder lstm's previous hTOut
	ptr<cType> hTIn;

private:
	ptr<cType> comp_LenXUnits;
};
class attTSBpg : public modelBpg, public vector<ptr<model>> {
public:
	RECBPGFIELDS
	attTSBpg();
	attTSBpg(int _xUnits, int _hTUnits);
	attTSBpg(int _xUnits, int _hTUnits, ptr<model> _seqTemplate);

	int xUnits;
	int hTUnits;
	// models that have been instantiated in RAM, and therefore are ready to be unrolled when ready to use
	vector<ptr<model>> prepared;
	// the template model is a sequential (using the attTSBpg(int, int) constructor, this will be a traditional neural network)
	ptr<model> seqTemplate;
	// the input hidden state is a vector, because it intakes one vector representing the decoder lstm's previous hTOut
	ptr<cType> hTIn;
	ptr<cType> hTInGrad;

private:
	ptr<cType> comp_LenXUnits;
	ptr<cType> comp_LenHTUnits;
};
class att : public model, public vector<ptr<model>> {
public:
	RECFIELDS
	att();
	att(int _xUnits, int _hTUnits);
	att(int _xUnits, int _hTUnits, ptr<model> _attTSTemplate);

	int xUnits;
	int hTUnits;

	// models that have been instantiated in RAM, and therefore are ready to be unrolled when ready to use
	vector<ptr<model>> prepared;
	// the template model is an attTS (attention timestep)
	ptr<model> attTSTemplate;
	// the input hidden state is a matrix, because it intakes one vector representing the decoder lstm's hidden state at every timestep
	ptr<cType> hTIn;
};
class attBpg : public modelBpg, public vector<ptr<model>> {
public:
	RECBPGFIELDS
	attBpg();
	attBpg(int _xUnits, int _hTUnits);
	attBpg(int _xUnits, int _hTUnits, ptr<model> _attTSTemplate);

	int xUnits;
	int hTUnits;
	// models that have been instantiated in RAM, and therefore are ready to be unrolled when ready to use
	vector<ptr<model>> prepared;
	// the template model is an attTS (attention timestep)
	ptr<model> attTSTemplate;
	// the input hidden state is a matrix, because it intakes one vector representing the decoder lstm's hidden state at every timestep
	ptr<cType> hTIn;
	ptr<cType> hTInGrad;
};