#pragma once
#include "superHeader.h"
#include "general.h"
#include <any>

class cType;
class cType {
public:
	cType();
	cType(double a);
	cType(vector<sPtr<cType>> a);
	cType(initializer_list<cType> a);
	double vDouble;
	vector<sPtr<cType>> vVector;
};

void add0D(sPtr<cType> a, sPtr<cType> b, sPtr<cType> output);
void add1D(sPtr<cType> a, sPtr<cType> b, sPtr<cType> output);

void sub0D(sPtr<cType> a, sPtr<cType> b, sPtr<cType> output);
void sub1D(sPtr<cType> a, sPtr<cType> b, sPtr<cType> output);

void mult0D(sPtr<cType> a, sPtr<cType> b, sPtr<cType> output);
void mult1D(sPtr<cType> a, sPtr<cType> b, sPtr<cType> output);

void div0D(sPtr<cType> a, sPtr<cType> b, sPtr<cType> output);
void div1D(sPtr<cType> a, sPtr<cType> b, sPtr<cType> output);