#pragma once
#include "maths.h"

cType::cType() {
	// initializes vDouble
	vDouble = 0;
}
cType::cType(double a) {
	vDouble = a;
}
cType::cType(vector<sPtr<cType>> a) {
	// initializes vDouble
	vDouble = 0;
	vVector = a;
}
cType::cType(initializer_list<cType> a) {
	// initializes vDouble
	vDouble = 0;

	vector<cType> vA = vector<cType>();
	copy(a.begin(), a.end(), back_inserter(vA));
	vVector = vector<sPtr<cType>>();
	for (cType c : vA) {
		vVector.push_back(new cType(c));
	}
}

void clear0D(sPtr<cType> a) {

	a->vDouble = 0;

}
void clear1D(sPtr<cType> a) {
	vector<sPtr<cType>>* vVec = &a->vVector;
	for (int i = 0; i < vVec->size(); i++) {
		clear0D(vVec->at(i));
	}
}
void clear2D(sPtr<cType> a) {
	vector<sPtr<cType>>* vVec = &a->vVector;
	for (int i = 0; i < vVec->size(); i++) {
		clear1D(vVec->at(i));
	}
}

void add0D(sPtr<cType> a, sPtr<cType> b, sPtr<cType> output) {
	output->vDouble = a->vDouble + b->vDouble;
}
void add1D(sPtr<cType> a, sPtr<cType> b, sPtr<cType> output) {

	// pull in reference to save compute
	vector<sPtr<cType>>* aVec = &a->vVector;
	vector<sPtr<cType>>* bVec = &b->vVector;
	vector<sPtr<cType>>* outVec = &output->vVector;
	
	// throw exception if vectors are of inequal sizes
	assert(aVec->size() == bVec->size() && aVec->size() == outVec->size());

	for (int i = 0; i < aVec->size(); i++) {
		outVec->at(i)->vDouble = aVec->at(i)->vDouble + bVec->at(i)->vDouble;
	}

}
void add2D(sPtr<cType> a, sPtr<cType> b, sPtr<cType> output) {

	// pull in reference to save compute
	vector<sPtr<cType>>* aVec = &a->vVector;
	vector<sPtr<cType>>* bVec = &b->vVector;
	vector<sPtr<cType>>* outVec = &output->vVector;

	// throw exception if vectors are of inequal sizes
	assert(aVec->size() == bVec->size() && aVec->size() == outVec->size());

	for (int i = 0; i < aVec->size(); i++) {
		add1D(aVec->at(i), bVec->at(i), outVec->at(i));
	}

}

sPtr<cType> add0D(sPtr<cType> a, sPtr<cType> b) {

	sPtr<cType> result = new cType();

	add0D(a, b, result);
	
	return result;
}
sPtr<cType> add1D(sPtr<cType> a, sPtr<cType> b) {

	vector<sPtr<cType>>* aVec = &a->vVector;
	vector<sPtr<cType>>* bVec = &b->vVector;
	sPtr<cType> result = new cType({});

	for (int i = 0; i < aVec->size(); i++) {
		result->vVector.push_back(add0D(aVec->at(i), bVec->at(i)));
	}

	return result;
}
sPtr<cType> add2D(sPtr<cType> a, sPtr<cType> b) {

	vector<sPtr<cType>>* aVec = &a->vVector;
	vector<sPtr<cType>>* bVec = &b->vVector;
	sPtr<cType> result = new cType({});

	for (int i = 0; i < aVec->size(); i++) {
		result->vVector.push_back(add1D(aVec->at(i), bVec->at(i)));
	}

	return result;
}

void sub0D(sPtr<cType> a, sPtr<cType> b, sPtr<cType> output) {
	output->vDouble = a->vDouble - b->vDouble;
}
void sub1D(sPtr<cType> a, sPtr<cType> b, sPtr<cType> output) {

	// pull in reference to save compute
	vector<sPtr<cType>>* aVec = &a->vVector;
	vector<sPtr<cType>>* bVec = &b->vVector;
	vector<sPtr<cType>>* outVec = &output->vVector;

	// throw exception if vectors are of inequal sizes
	assert(aVec->size() == bVec->size() && aVec->size() == outVec->size());

	for (int i = 0; i < aVec->size(); i++) {
		outVec->at(i)->vDouble = aVec->at(i)->vDouble - bVec->at(i)->vDouble;
	}

}
void sub2D(sPtr<cType> a, sPtr<cType> b, sPtr<cType> output) {
	// pull in reference to save compute
	vector<sPtr<cType>>* aVec = &a->vVector;
	vector<sPtr<cType>>* bVec = &b->vVector;
	vector<sPtr<cType>>* outVec = &output->vVector;

	// throw exception if vectors are of inequal sizes
	assert(aVec->size() == bVec->size() && aVec->size() == outVec->size());

	for (int i = 0; i < aVec->size(); i++) {
		sub1D(aVec->at(i), bVec->at(i), outVec->at(i));
	}
}

sPtr<cType> sub0D(sPtr<cType> a, sPtr<cType> b) {

	sPtr<cType> result = new cType();

	sub0D(a, b, result);

	return result;
}
sPtr<cType> sub1D(sPtr<cType> a, sPtr<cType> b) {

	vector<sPtr<cType>>* aVec = &a->vVector;
	vector<sPtr<cType>>* bVec = &b->vVector;
	sPtr<cType> result = new cType({});

	for (int i = 0; i < aVec->size(); i++) {
		result->vVector.push_back(sub0D(aVec->at(i), bVec->at(i)));
	}

	return result;
}
sPtr<cType> sub2D(sPtr<cType> a, sPtr<cType> b) {

	vector<sPtr<cType>>* aVec = &a->vVector;
	vector<sPtr<cType>>* bVec = &b->vVector;
	sPtr<cType> result = new cType({});

	for (int i = 0; i < aVec->size(); i++) {
		result->vVector.push_back(sub1D(aVec->at(i), bVec->at(i)));
	}

	return result;
}

void mult0D(sPtr<cType> a, sPtr<cType> b, sPtr<cType> output) {
	output->vDouble = a->vDouble * b->vDouble;
}
void mult1D(sPtr<cType> a, sPtr<cType> b, sPtr<cType> output) {

	// pull in reference to save compute
	vector<sPtr<cType>>* aVec = &a->vVector;
	vector<sPtr<cType>>* bVec = &b->vVector;
	vector<sPtr<cType>>* outVec = &output->vVector;

	// throw exception if vectors are of inequal sizes
	assert(aVec->size() == bVec->size() && aVec->size() == outVec->size());

	for (int i = 0; i < aVec->size(); i++) {
		outVec->at(i)->vDouble = aVec->at(i)->vDouble * bVec->at(i)->vDouble;
	}

}
void mult2D(sPtr<cType> a, sPtr<cType> b, sPtr<cType> output) {

	// pull in reference to save compute
	vector<sPtr<cType>>* aVec = &a->vVector;
	vector<sPtr<cType>>* bVec = &b->vVector;
	vector<sPtr<cType>>* outVec = &output->vVector;

	// throw exception if vectors are of inequal sizes
	assert(aVec->size() == bVec->size() && aVec->size() == outVec->size());

	for (int i = 0; i < aVec->size(); i++) {
		mult1D(aVec->at(i), bVec->at(i), outVec->at(i));
	}

}

sPtr<cType> mult0D(sPtr<cType> a, sPtr<cType> b) {

	sPtr<cType> result = new cType();

	mult0D(a, b, result);

	return result;
}
sPtr<cType> mult1D(sPtr<cType> a, sPtr<cType> b) {

	vector<sPtr<cType>>* aVec = &a->vVector;
	vector<sPtr<cType>>* bVec = &b->vVector;
	sPtr<cType> result = new cType({});

	for (int i = 0; i < aVec->size(); i++) {
		result->vVector.push_back(mult0D(aVec->at(i), bVec->at(i)));
	}

	return result;
}
sPtr<cType> mult2D(sPtr<cType> a, sPtr<cType> b) {

	vector<sPtr<cType>>* aVec = &a->vVector;
	vector<sPtr<cType>>* bVec = &b->vVector;
	sPtr<cType> result = new cType({});

	for (int i = 0; i < aVec->size(); i++) {
		result->vVector.push_back(mult1D(aVec->at(i), bVec->at(i)));
	}

	return result;
}

void div0D(sPtr<cType> a, sPtr<cType> b, sPtr<cType> output) {
	output->vDouble = a->vDouble / b->vDouble;
}
void div1D(sPtr<cType> a, sPtr<cType> b, sPtr<cType> output) {

	// pull in reference to save compute
	vector<sPtr<cType>>* aVec = &a->vVector;
	vector<sPtr<cType>>* bVec = &b->vVector;
	vector<sPtr<cType>>* outVec = &output->vVector;

	// throw exception if vectors are of inequal sizes
	assert(aVec->size() == bVec->size() && aVec->size() == outVec->size());

	for (int i = 0; i < aVec->size(); i++) {
		outVec->at(i)->vDouble = aVec->at(i)->vDouble / bVec->at(i)->vDouble;
	}

}
void div2D(sPtr<cType> a, sPtr<cType> b, sPtr<cType> output) {

	// pull in reference to save compute
	vector<sPtr<cType>>* aVec = &a->vVector;
	vector<sPtr<cType>>* bVec = &b->vVector;
	vector<sPtr<cType>>* outVec = &output->vVector;

	// throw exception if vectors are of inequal sizes
	assert(aVec->size() == bVec->size() && aVec->size() == outVec->size());

	for (int i = 0; i < aVec->size(); i++) {
		div1D(aVec->at(i), bVec->at(i), outVec->at(i));
	}

}

sPtr<cType> div0D(sPtr<cType> a, sPtr<cType> b) {

	sPtr<cType> result = new cType();

	div0D(a, b, result);

	return result;
}
sPtr<cType> div1D(sPtr<cType> a, sPtr<cType> b) {

	vector<sPtr<cType>>* aVec = &a->vVector;
	vector<sPtr<cType>>* bVec = &b->vVector;
	sPtr<cType> result = new cType({});

	for (int i = 0; i < aVec->size(); i++) {
		result->vVector.push_back(div0D(aVec->at(i), bVec->at(i)));
	}

	return result;
}
sPtr<cType> div2D(sPtr<cType> a, sPtr<cType> b) {

	vector<sPtr<cType>>* aVec = &a->vVector;
	vector<sPtr<cType>>* bVec = &b->vVector;
	sPtr<cType> result = new cType({});

	for (int i = 0; i < aVec->size(); i++) {
		result->vVector.push_back(div1D(aVec->at(i), bVec->at(i)));
	}

	return result;
}

void abs0D(sPtr<cType> a, sPtr<cType> output) {
	output->vDouble = abs(a->vDouble);
}
void abs1D(sPtr<cType> a, sPtr<cType> output) {

	// pull in reference to save compute
	vector<sPtr<cType>>* aVec = &a->vVector;
	vector<sPtr<cType>>* outVec = &output->vVector;

	for (int i = 0; i < aVec->size(); i++) {
		outVec->at(i)->vDouble = abs(aVec->at(i)->vDouble);
	}
}
void abs2D(sPtr<cType> a, sPtr<cType> output) {

	// pull in reference to save compute
	vector<sPtr<cType>>* aVec = &a->vVector;
	vector<sPtr<cType>>* outVec = &output->vVector;

	for (int i = 0; i < aVec->size(); i++) {
		abs1D(aVec->at(i), outVec->at(i));
	}
}

sPtr<cType> abs0D(sPtr<cType> a) {
	sPtr<cType> result = new cType(0);
	abs0D(a, result);
	return result;
}
sPtr<cType> abs1D(sPtr<cType> a) {
	// pull in reference to save compute
	vector<sPtr<cType>>* aVec = &a->vVector;

	sPtr<cType> result = new cType({});
	for (int i = 0; i < aVec->size(); i++) {
		result->vVector.push_back(new cType());
	}
	abs1D(a, result);
	return result;
}
sPtr<cType> abs2D(sPtr<cType> a) {
	sPtr<cType> result = new cType({});

	// pull in reference to save compute
	vector<sPtr<cType>>* aVec = &a->vVector;

	for (int i = 0; i < aVec->size(); i++) {
		result->vVector.push_back(abs1D(aVec->at(i)));
	}
	return result;
}

void sum1D(sPtr<cType> a, sPtr<cType> output) {

	// pull in reference to save compute
	vector<sPtr<cType>>* aVec = &a->vVector;

	double result = 0;

	for (int i = 0; i < aVec->size(); i++) {
		result += aVec->at(i)->vDouble;
	}

	output->vDouble = result;

}
void sum2D(sPtr<cType> a, sPtr<cType> output) {

	// pull in reference to save compute
	vector<sPtr<cType>>* aVec = &a->vVector;
	vector<sPtr<cType>>* outVec = &a->vVector;

	for (int i = 0; i < aVec->size(); i++) {
		sum1D(aVec->at(i), outVec->at(i));
	}

}
sPtr<cType> sum1D(sPtr<cType> a) {
	sPtr<cType> result = new cType();
	sum1D(a, result);
	return result;
}
sPtr<cType> sum2D(sPtr<cType> a) {
	sPtr<cType> result = new cType({});

	// pull in reference to save compute
	vector<sPtr<cType>>* aVec = &a->vVector;

	for (int i = 0; i < aVec->size(); i++) {
		result->vVector.push_back(sum1D(aVec->at(i)));
	}
	return result;
}

void concat(sPtr<cType> a, sPtr<cType> b, sPtr<cType> output) {

	// pull in reference to save compute
	vector<sPtr<cType>>* aVec = &a->vVector;
	vector<sPtr<cType>>* bVec = &b->vVector;
	vector<sPtr<cType>>* outVec = &output->vVector;

	int index = 0;
	for (int i = 0; i < aVec->size(); i++) {
		outVec->at(index) = aVec->at(i);
		index++;
	}
	for (int i = 0; i < bVec->size(); i++) {
		outVec->at(index) = bVec->at(i);
		index++;
	}
}
sPtr<cType> concat(sPtr<cType> a, sPtr<cType> b) {

	vector<sPtr<cType>>* aVec = &a->vVector;
	vector<sPtr<cType>>* bVec = &b->vVector;

	sPtr<cType> result = new cType({});

	for (int i = 0; i < aVec->size(); i++) {
		result->vVector.push_back(new cType());
	}
	for (int i = 0; i < bVec->size(); i++) {
		result->vVector.push_back(new cType());
	}

	concat(a, b, result);
	return result;

}

sPtr<vector<int>> randomDist(int count, int incMin, int excMax, bool replace) {

	vector<int> allPossibilities = vector<int>();
	for (int i = incMin; i < excMax; i++) {
		allPossibilities.push_back(i);
	}

	sPtr<vector<int>> result = new vector<int>();
	for (int i = 0; i < count; i++) {
		int choiceIndex = rand() % allPossibilities.size();
		result->push_back(allPossibilities.at(choiceIndex));

		// remove choice from possible choices if replace is set to false
		if (!replace) {
			allPossibilities.erase(allPossibilities.begin() + choiceIndex);
		}
	}

	return result;

}

double actFunc::eval(double x) {
	return x;
}
double actFunc::deriv(double y) {
	return 1;
}

double actFuncSm::eval(double x) {
	return 1 / (1 + exp(-x));
}
double actFuncSm::deriv(double y) {
	return y * (1 - y);
}

double actFuncTh::eval(double x) {
	return tanh(x);
}
double actFuncTh::deriv(double y) {
	return 1 / (pow(cosh(y), 2));
}

actFuncLR::actFuncLR(double m) {
	this->m = m;
}
double actFuncLR::eval(double x) {
	if (x > 0) {
		return x;
	}
	else {
		return m * x;
	}
}
double actFuncLR::deriv(double y) {
	if (y > 0) {
		return 1;
	}
	else {
		return m;
	}
}