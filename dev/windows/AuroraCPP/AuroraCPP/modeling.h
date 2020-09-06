#pragma once
#include "superHeader.h"
#include "optimization.h"
#include "maths.h"
#include "general.h"

#pragma region Defs

#define MODELFIELDS \
virtual void fwd();\
virtual void bwd();\
virtual ptr<model> clone();

#define SEQFIELDS MODELFIELDS \
virtual void modelWise(function<void(model*)> func);

#define RECFIELDS SEQFIELDS \
virtual void incFwd(int a);\
virtual void incBwd(int a);\
virtual void prep(int a);\
virtual void unroll(int a);\
virtual void clear(); \
int index;

#define ATTFIELDS \
RECFIELDS \
virtual void prep(int a, int b);\
virtual void unroll(int a, int b);


#pragma endregion

class model;
class bias;
class act;
class weight;
class wSet;
class wJunc;
class seq;
class layer;
class sync;
class lstmTS;
class lstm;
class muTS;
class mu;
class attTS;
class att;
class cnl;
class cnn;

seq* tnn(vector<int> npl, vector<ptr<model>> layerNeuronTemplates);
seq* tnn(vector<int> npl, ptr<model> neuronTemplate);
seq* cnn(ptr<model> _cnl, int _cnls);
seq* cnn(int _a, int _b, int _cnls);
seq* cnn(vector<ptr<model>> _cnls);

//seq* stackedLstm(int count, int units);
//seq* stackedlstm(int count, int units);

seq* neuronSm();
seq* neuronTh();
seq* neuronLR(double m);

void extractParam(model* m, vector<ptr<param>*>& paramVecOutput);
void extractParams(model* m, vector<ptr<param>*>& paramVecOutput);
void extractParams(ptr<model> m, vector<ptr<param>*>& paramVecOutput);

//void attToLSTMFwd(att* a, lstm* l);
//void attToLSTMFwd(att* a, lstm* l);
//void attToLSTMBwd(att* a, lstm* l);

class model {
public:
	model();
	virtual void fwd();
	virtual void bwd();
	virtual void modelWise(function<void(model*)> func);
	virtual ptr<model> clone();
	ptr<cType> x;
	ptr<cType> y;
	ptr<cType> xGrad;
	ptr<cType> yGrad;
};
class bias : public model {
public:
	MODELFIELDS
	bias();
	ptr<param> prm;
};
class act : public model {
public:
	MODELFIELDS
	act();
	act(actFunc* _af);
	act(ptr<actFunc> _af);
	ptr<actFunc> af;
};
class weight : public model {
public:
	MODELFIELDS
	weight();
	ptr<param> prm;
};
class wSet : public model, public vector<ptr<model>> {
public:
	SEQFIELDS
	wSet();
	wSet(int _a);
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
class seq : public model, public vector<ptr<model>> {
public:
	SEQFIELDS
	seq();
	seq(vector<ptr<model>>& _models);
	seq(initializer_list<ptr<model>> _models);
};
class layer : public model, public vector<ptr<model>> {
public:
	SEQFIELDS
	layer();
	layer(int a, model* modelTemplate);
	layer(int a, ptr<model> modelTemplate);
	layer(initializer_list<ptr<model>> _models);
};
class sync : public model, public vector<ptr<model>> {
public:
	RECFIELDS
	sync();
	sync(model* _modelTemplate);
	sync(ptr<model> _modelTemplate);

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
	//lstm(int _units, vector<int> gateHiddenLayers);
	//lstm(int _units, vector<int> _aGateHiddenLayers, vector<int> _bGateHiddenLayers, vector<int> _cGateHiddenLayers, vector<int> _dGateHiddenLayers);
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
	//mu(int _xUnits, int _cTUnits, int _hTUnits, vector<int> gateHiddenLayers);
	mu(int _xUnits, int _cTUnits, int _hTUnits, ptr<model> gate);

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
	// the template model is a sequential (using the attTS(int, int) constructor, this will be a traditional neural network)
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
	ATTFIELDS
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
	ptr<cType> hTInGrad;
};
class cnl : public model, public vector<ptr<model>> {
public:
	SEQFIELDS
	cnl();
	cnl(int _a, int _b);
	cnl(int _a, int _b, ptr<model> _filterTemplate);
	virtual void prep(int a);
	virtual void unroll(int a);
	virtual void clear();

	int numSteps(int xSize);
	int ySize(int xSize);
	int xSize(int ySize);

	int a;
	int b;

	ptr<model> filterTemplate;

	// models that have been instantiated and are prepared for use, when unroll is called
	vector<ptr<model>> prepared;

private:
	ptr<cType> comp_LenA;
};