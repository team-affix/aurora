#pragma once
#include "superHeader.h"
#include "general.h"
#include <any>

class cType;
class actFunc;
class actFuncSm;
class actFuncTh;
class actFuncLR;

// default carry type for all models
class cType {
public:
	cType();
	cType(double a);
	cType(vector<sPtr<cType>> a);
	cType(initializer_list<cType> a);
	double vDouble;
	vector<sPtr<cType>> vVector;
};

void clear0D(sPtr<cType> a);
void clear1D(sPtr<cType> a);

void add0D(sPtr<cType> a, sPtr<cType> b, sPtr<cType> output);
void add1D(sPtr<cType> a, sPtr<cType> b, sPtr<cType> output);

void sub0D(sPtr<cType> a, sPtr<cType> b, sPtr<cType> output);
void sub1D(sPtr<cType> a, sPtr<cType> b, sPtr<cType> output);

void mult0D(sPtr<cType> a, sPtr<cType> b, sPtr<cType> output);
void mult1D(sPtr<cType> a, sPtr<cType> b, sPtr<cType> output);

void div0D(sPtr<cType> a, sPtr<cType> b, sPtr<cType> output);
void div1D(sPtr<cType> a, sPtr<cType> b, sPtr<cType> output);

// default activation function, just returns input
class actFunc {
public:
	virtual double eval(double x);
	virtual double deriv(double y);
};

// softmax activation function, return sigmoid of input
class actFuncSm : public actFunc {
public:
	virtual double eval(double x);
	virtual double deriv(double y);
};

// tanh activation function, return hyperbolic tangent of input
class actFuncTh : public actFunc {
public:
	virtual double eval(double x);
	virtual double deriv(double y);
};

// leaky relu activation function, returns leaky relu of input
class actFuncLR : public actFunc {
public:
	actFuncLR(double m);
	virtual double eval(double x);
	virtual double deriv(double y);
	double m;
};