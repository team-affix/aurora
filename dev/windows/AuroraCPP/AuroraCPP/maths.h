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
	cType(vector<ptr<cType>> a);
	cType(initializer_list<cType> a);
	double vDouble;
	vector<ptr<cType>> vVector;
};

ptr<cType> make1D(int a);
ptr<cType> make2D(int a, int b);
ptr<cType> make3D(int a, int b, int c);
ptr<cType> make4D(int a, int b, int c, int d);
ptr<cType> make5D(int a, int b, int c, int d, int e);

void clear0D(ptr<cType> a);
void clear1D(ptr<cType> a);
void clear2D(ptr<cType> a);

void copy0D(ptr<cType> a, ptr<cType> output);
void copy1D(ptr<cType> a, ptr<cType> output);
void copy1D(ptr<cType> a, ptr<cType> output, int sourceStartIndex, int count, int destStartIndex);
void copy2D(ptr<cType> a, ptr<cType> output);
void copy2D(ptr<cType> a, ptr<cType> output, int sourceStartIndex, int count, int destStartIndex);

void add0D(ptr<cType> a, ptr<cType> b, ptr<cType> output);
void add1D(ptr<cType> a, ptr<cType> b, ptr<cType> output);
void add2D(ptr<cType> a, ptr<cType> b, ptr<cType> output);

ptr<cType> add0D(ptr<cType> a, ptr<cType> b);
ptr<cType> add1D(ptr<cType> a, ptr<cType> b);
ptr<cType> add2D(ptr<cType> a, ptr<cType> b);

void sub0D(ptr<cType> a, ptr<cType> b, ptr<cType> output);
void sub1D(ptr<cType> a, ptr<cType> b, ptr<cType> output);
void sub2D(ptr<cType> a, ptr<cType> b, ptr<cType> output);

ptr<cType> sub0D(ptr<cType> a, ptr<cType> b);
ptr<cType> sub1D(ptr<cType> a, ptr<cType> b);
ptr<cType> sub2D(ptr<cType> a, ptr<cType> b);

void mult0D(ptr<cType> a, ptr<cType> b, ptr<cType> output);
void mult1D(ptr<cType> a, ptr<cType> b, ptr<cType> output);
void mult2D(ptr<cType> a, ptr<cType> b, ptr<cType> output);

ptr<cType> mult0D(ptr<cType> a, ptr<cType> b);
ptr<cType> mult1D(ptr<cType> a, ptr<cType> b);
ptr<cType> mult2D(ptr<cType> a, ptr<cType> b);

void div0D(ptr<cType> a, ptr<cType> b, ptr<cType> output);
void div1D(ptr<cType> a, ptr<cType> b, ptr<cType> output);
void div2D(ptr<cType> a, ptr<cType> b, ptr<cType> output);

ptr<cType> div0D(ptr<cType> a, ptr<cType> b);
ptr<cType> div1D(ptr<cType> a, ptr<cType> b);
ptr<cType> div2D(ptr<cType> a, ptr<cType> b);

void abs0D(ptr<cType> a, ptr<cType> output);
void abs1D(ptr<cType> a, ptr<cType> output);
void abs2D(ptr<cType> a, ptr<cType> output);

ptr<cType> abs0D(ptr<cType> a);
ptr<cType> abs1D(ptr<cType> a);
ptr<cType> abs2D(ptr<cType> a);

void sum1D(ptr<cType> a, ptr<cType> output);
void sum2D(ptr<cType> a, ptr<cType> output);
ptr<cType> sum1D(ptr<cType> a);
ptr<cType> sum2D(ptr<cType> a);

void concat(ptr<cType> a, ptr<cType> b, ptr<cType> output);
void concat(vector<ptr<cType>> vec, ptr<cType> output);
ptr<cType> concat(ptr<cType> a, ptr<cType> b);

// default activation function, just returns input
class actFunc {
public:
	virtual double eval(double x);
	virtual double deriv(double* x, double* y);
};

// softmax activation function, return sigmoid of input
class actFuncSm : public actFunc {
public:
	virtual double eval(double x);
	virtual double deriv(double* x, double* y);
};

// tanh activation function, return hyperbolic tangent of input
class actFuncTh : public actFunc {
public:
	virtual double eval(double x);
	virtual double deriv(double* x, double* y);
};

// leaky relu activation function, returns leaky relu of input
class actFuncLR : public actFunc {
public:
	actFuncLR(double m);
	virtual double eval(double x);
	virtual double deriv(double* x, double* y);
	double m;
};
