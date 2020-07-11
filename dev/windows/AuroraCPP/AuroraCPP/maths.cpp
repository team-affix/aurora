#pragma once
#include "maths.h"

cType::cType() {
	// initializes vDouble
	vDouble = 0;
}
cType::cType(double a) {
	vDouble = a;
}
cType::cType(vector<ptr<cType>> a) {
	// initializes vDouble
	vDouble = 0;
	vVector = a;
}
cType::cType(initializer_list<cType> a) {
	// initializes vDouble
	vDouble = 0;

	vector<cType> vA = vector<cType>();
	copy(a.begin(), a.end(), back_inserter(vA));
	vVector = vector<ptr<cType>>();
	for (cType c : vA) {
		vVector.push_back(new cType(c));
	}
}

ptr<cType> make1D(int a) {
	cType* result = new cType({});
	for (int i = 0; i < a; i++) {
		result->vVector.push_back(new cType(0));
	}
	return result;
}
ptr<cType> make2D(int a, int b) {
	cType* result = new cType({});
	for (int i = 0; i < a; i++) {
		result->vVector.push_back(make1D(b));
	}
	return result;
}
ptr<cType> make3D(int a, int b, int c) {
	cType* result = new cType({});
	for (int i = 0; i < a; i++) {
		result->vVector.push_back(make2D(b, c));
	}
	return result;
}
ptr<cType> make4D(int a, int b, int c, int d) {
	cType* result = new cType({});
	for (int i = 0; i < a; i++) {
		result->vVector.push_back(make3D(b, c, d));
	}
	return result;
}
ptr<cType> make5D(int a, int b, int c, int d, int e) {
	cType* result = new cType({});
	for (int i = 0; i < a; i++) {
		result->vVector.push_back(make4D(b, c, d, e));
	}
	return result;
}

void clear0D(ptr<cType> a) {

	a->vDouble = 0;

}
void clear1D(ptr<cType> a) {
	vector<ptr<cType>>* vVec = &a->vVector;
	for (int i = 0; i < vVec->size(); i++) {
		clear0D(vVec->at(i));
	}
}
void clear2D(ptr<cType> a) {
	vector<ptr<cType>>* vVec = &a->vVector;
	for (int i = 0; i < vVec->size(); i++) {
		clear1D(vVec->at(i));
	}
}

void copy0D(ptr<cType> a, ptr<cType> output) {
	output->vDouble = a->vDouble;
}
void copy1D(ptr<cType> a, ptr<cType> output) {

	// pull in reference to save compute
	vector<ptr<cType>>* aVec = &a->vVector;
	vector<ptr<cType>>* outVec = &output->vVector;

	// throw exception if vectors are of inequal sizes
	assert(aVec->size() == outVec->size());

	for (int i = 0; i < aVec->size(); i++) {
		copy0D(aVec->at(i), outVec->at(i));
	}

}
void copy1D(ptr<cType> a, ptr<cType> output, int sourceStartIndex, int count, int destStartIndex) {

	// pull in reference to save compute
	vector<ptr<cType>>* aVec = &a->vVector;
	vector<ptr<cType>>* outVec = &output->vVector;

	// throw exception if vectors are of inequal sizes
	assert(aVec->size() >= sourceStartIndex + count || outVec->size() >= destStartIndex + count);

	for (int i = 0; i < count; i++) {
		int sourceIndex = sourceStartIndex + i;
		int destIndex = destStartIndex + i;
		copy0D(aVec->at(sourceIndex), outVec->at(destIndex));
	}

}
void copy2D(ptr<cType> a, ptr<cType> output) {

	// pull in reference to save compute
	vector<ptr<cType>>* aVec = &a->vVector;
	vector<ptr<cType>>* outVec = &output->vVector;

	// throw exception if vectors are of inequal sizes
	assert(aVec->size() == outVec->size());

	for (int i = 0; i < aVec->size(); i++) {
		copy1D(aVec->at(i), outVec->at(i));
	}

}
void copy2D(ptr<cType> a, ptr<cType> output, int sourceStartIndex, int count, int destStartIndex) {

	// pull in reference to save compute
	vector<ptr<cType>>* aVec = &a->vVector;
	vector<ptr<cType>>* outVec = &output->vVector;

	// throw exception if vectors are of inequal sizes
	assert(aVec->size() == outVec->size());

	for (int i = 0; i < count; i++) {
		int sourceIndex = sourceStartIndex + i;
		int destIndex = destStartIndex + i;
		copy1D(aVec->at(sourceIndex), outVec->at(destIndex));
	}

}

void add0D(ptr<cType> a, ptr<cType> b, ptr<cType> output) {
	output->vDouble = a->vDouble + b->vDouble;
}
void add1D(ptr<cType> a, ptr<cType> b, ptr<cType> output) {

	// pull in reference to save compute
	vector<ptr<cType>>* aVec = &a->vVector;
	vector<ptr<cType>>* bVec = &b->vVector;
	vector<ptr<cType>>* outVec = &output->vVector;
	
	// throw exception if vectors are of inequal sizes
	assert(aVec->size() == bVec->size() && aVec->size() == outVec->size());

	for (int i = 0; i < aVec->size(); i++) {
		add0D(aVec->at(i), bVec->at(i), outVec->at(i));
	}

}
void add2D(ptr<cType> a, ptr<cType> b, ptr<cType> output) {

	// pull in reference to save compute
	vector<ptr<cType>>* aVec = &a->vVector;
	vector<ptr<cType>>* bVec = &b->vVector;
	vector<ptr<cType>>* outVec = &output->vVector;

	// throw exception if vectors are of inequal sizes
	assert(aVec->size() == bVec->size() && aVec->size() == outVec->size());

	for (int i = 0; i < aVec->size(); i++) {
		add1D(aVec->at(i), bVec->at(i), outVec->at(i));
	}

}

ptr<cType> add0D(ptr<cType> a, ptr<cType> b) {

	ptr<cType> result = new cType();

	add0D(a, b, result);
	
	return result;
}
ptr<cType> add1D(ptr<cType> a, ptr<cType> b) {

	vector<ptr<cType>>* aVec = &a->vVector;
	vector<ptr<cType>>* bVec = &b->vVector;
	ptr<cType> result = new cType({});

	for (int i = 0; i < aVec->size(); i++) {
		result->vVector.push_back(add0D(aVec->at(i), bVec->at(i)));
	}

	return result;
}
ptr<cType> add2D(ptr<cType> a, ptr<cType> b) {

	vector<ptr<cType>>* aVec = &a->vVector;
	vector<ptr<cType>>* bVec = &b->vVector;
	ptr<cType> result = new cType({});

	for (int i = 0; i < aVec->size(); i++) {
		result->vVector.push_back(add1D(aVec->at(i), bVec->at(i)));
	}

	return result;
}

void sub0D(ptr<cType> a, ptr<cType> b, ptr<cType> output) {
	output->vDouble = a->vDouble - b->vDouble;
}
void sub1D(ptr<cType> a, ptr<cType> b, ptr<cType> output) {

	// pull in reference to save compute
	vector<ptr<cType>>* aVec = &a->vVector;
	vector<ptr<cType>>* bVec = &b->vVector;
	vector<ptr<cType>>* outVec = &output->vVector;

	// throw exception if vectors are of inequal sizes
	assert(aVec->size() == bVec->size() && aVec->size() == outVec->size());

	for (int i = 0; i < aVec->size(); i++) {
		outVec->at(i)->vDouble = aVec->at(i)->vDouble - bVec->at(i)->vDouble;
	}

}
void sub2D(ptr<cType> a, ptr<cType> b, ptr<cType> output) {
	// pull in reference to save compute
	vector<ptr<cType>>* aVec = &a->vVector;
	vector<ptr<cType>>* bVec = &b->vVector;
	vector<ptr<cType>>* outVec = &output->vVector;

	// throw exception if vectors are of inequal sizes
	assert(aVec->size() == bVec->size() && aVec->size() == outVec->size());

	for (int i = 0; i < aVec->size(); i++) {
		sub1D(aVec->at(i), bVec->at(i), outVec->at(i));
	}
}

ptr<cType> sub0D(ptr<cType> a, ptr<cType> b) {

	ptr<cType> result = new cType();

	sub0D(a, b, result);

	return result;
}
ptr<cType> sub1D(ptr<cType> a, ptr<cType> b) {

	vector<ptr<cType>>* aVec = &a->vVector;
	vector<ptr<cType>>* bVec = &b->vVector;
	ptr<cType> result = new cType({});

	for (int i = 0; i < aVec->size(); i++) {
		result->vVector.push_back(sub0D(aVec->at(i), bVec->at(i)));
	}

	return result;
}
ptr<cType> sub2D(ptr<cType> a, ptr<cType> b) {

	vector<ptr<cType>>* aVec = &a->vVector;
	vector<ptr<cType>>* bVec = &b->vVector;
	ptr<cType> result = new cType({});

	for (int i = 0; i < aVec->size(); i++) {
		result->vVector.push_back(sub1D(aVec->at(i), bVec->at(i)));
	}

	return result;
}

void mult0D(ptr<cType> a, ptr<cType> b, ptr<cType> output) {
	output->vDouble = a->vDouble * b->vDouble;
}
void mult1D(ptr<cType> a, ptr<cType> b, ptr<cType> output) {

	// pull in reference to save compute
	vector<ptr<cType>>* aVec = &a->vVector;
	vector<ptr<cType>>* bVec = &b->vVector;
	vector<ptr<cType>>* outVec = &output->vVector;

	// throw exception if vectors are of inequal sizes
	assert(aVec->size() == bVec->size() && aVec->size() == outVec->size());

	for (int i = 0; i < aVec->size(); i++) {
		mult0D(aVec->at(i), bVec->at(i), outVec->at(i));
	}

}
void mult2D(ptr<cType> a, ptr<cType> b, ptr<cType> output) {

	// pull in reference to save compute
	vector<ptr<cType>>* aVec = &a->vVector;
	vector<ptr<cType>>* bVec = &b->vVector;
	vector<ptr<cType>>* outVec = &output->vVector;

	// throw exception if vectors are of inequal sizes
	assert(aVec->size() == bVec->size() && aVec->size() == outVec->size());

	for (int i = 0; i < aVec->size(); i++) {
		mult1D(aVec->at(i), bVec->at(i), outVec->at(i));
	}

}

ptr<cType> mult0D(ptr<cType> a, ptr<cType> b) {

	ptr<cType> result = new cType();

	mult0D(a, b, result);

	return result;
}
ptr<cType> mult1D(ptr<cType> a, ptr<cType> b) {

	vector<ptr<cType>>* aVec = &a->vVector;
	vector<ptr<cType>>* bVec = &b->vVector;
	ptr<cType> result = new cType({});

	for (int i = 0; i < aVec->size(); i++) {
		result->vVector.push_back(mult0D(aVec->at(i), bVec->at(i)));
	}

	return result;
}
ptr<cType> mult2D(ptr<cType> a, ptr<cType> b) {

	vector<ptr<cType>>* aVec = &a->vVector;
	vector<ptr<cType>>* bVec = &b->vVector;
	ptr<cType> result = new cType({});

	for (int i = 0; i < aVec->size(); i++) {
		result->vVector.push_back(mult1D(aVec->at(i), bVec->at(i)));
	}

	return result;
}

void div0D(ptr<cType> a, ptr<cType> b, ptr<cType> output) {
	output->vDouble = a->vDouble / b->vDouble;
}
void div1D(ptr<cType> a, ptr<cType> b, ptr<cType> output) {

	// pull in reference to save compute
	vector<ptr<cType>>* aVec = &a->vVector;
	vector<ptr<cType>>* bVec = &b->vVector;
	vector<ptr<cType>>* outVec = &output->vVector;

	// throw exception if vectors are of inequal sizes
	assert(aVec->size() == bVec->size() && aVec->size() == outVec->size());

	for (int i = 0; i < aVec->size(); i++) {
		div0D(aVec->at(i), bVec->at(i), outVec->at(i));
	}

}
void div2D(ptr<cType> a, ptr<cType> b, ptr<cType> output) {

	// pull in reference to save compute
	vector<ptr<cType>>* aVec = &a->vVector;
	vector<ptr<cType>>* bVec = &b->vVector;
	vector<ptr<cType>>* outVec = &output->vVector;

	// throw exception if vectors are of inequal sizes
	assert(aVec->size() == bVec->size() && aVec->size() == outVec->size());

	for (int i = 0; i < aVec->size(); i++) {
		div1D(aVec->at(i), bVec->at(i), outVec->at(i));
	}

}

ptr<cType> div0D(ptr<cType> a, ptr<cType> b) {

	ptr<cType> result = new cType();

	div0D(a, b, result);

	return result;
}
ptr<cType> div1D(ptr<cType> a, ptr<cType> b) {

	vector<ptr<cType>>* aVec = &a->vVector;
	vector<ptr<cType>>* bVec = &b->vVector;
	ptr<cType> result = new cType({});

	for (int i = 0; i < aVec->size(); i++) {
		result->vVector.push_back(div0D(aVec->at(i), bVec->at(i)));
	}

	return result;
}
ptr<cType> div2D(ptr<cType> a, ptr<cType> b) {

	vector<ptr<cType>>* aVec = &a->vVector;
	vector<ptr<cType>>* bVec = &b->vVector;
	ptr<cType> result = new cType({});

	for (int i = 0; i < aVec->size(); i++) {
		result->vVector.push_back(div1D(aVec->at(i), bVec->at(i)));
	}

	return result;
}

void abs0D(ptr<cType> a, ptr<cType> output) {
	output->vDouble = abs(a->vDouble);
}
void abs1D(ptr<cType> a, ptr<cType> output) {

	// pull in reference to save compute
	vector<ptr<cType>>* aVec = &a->vVector;
	vector<ptr<cType>>* outVec = &output->vVector;

	for (int i = 0; i < aVec->size(); i++) {
		abs0D(aVec->at(i), outVec->at(i));
	}
}
void abs2D(ptr<cType> a, ptr<cType> output) {

	// pull in reference to save compute
	vector<ptr<cType>>* aVec = &a->vVector;
	vector<ptr<cType>>* outVec = &output->vVector;

	for (int i = 0; i < aVec->size(); i++) {
		abs1D(aVec->at(i), outVec->at(i));
	}
}

ptr<cType> abs0D(ptr<cType> a) {
	ptr<cType> result = new cType(0);
	abs0D(a, result);
	return result;
}
ptr<cType> abs1D(ptr<cType> a) {
	// pull in reference to save compute
	vector<ptr<cType>>* aVec = &a->vVector;

	ptr<cType> result = new cType({});
	for (int i = 0; i < aVec->size(); i++) {
		result->vVector.push_back(new cType());
	}
	abs1D(a, result);
	return result;
}
ptr<cType> abs2D(ptr<cType> a) {
	ptr<cType> result = new cType({});

	// pull in reference to save compute
	vector<ptr<cType>>* aVec = &a->vVector;

	for (int i = 0; i < aVec->size(); i++) {
		result->vVector.push_back(abs1D(aVec->at(i)));
	}
	return result;
}

void sum1D(ptr<cType> a, ptr<cType> output) {

	// pull in reference to save compute
	vector<ptr<cType>>* aVec = &a->vVector;

	double result = 0;

	for (int i = 0; i < aVec->size(); i++) {
		result += aVec->at(i)->vDouble;
	}

	output->vDouble = result;

}
void sum2D(ptr<cType> a, ptr<cType> output) {

	// pull in reference to save compute
	vector<ptr<cType>>* aVec = &a->vVector;
	vector<ptr<cType>>* outVec = &a->vVector;

	for (int i = 0; i < aVec->size(); i++) {
		sum1D(aVec->at(i), outVec->at(i));
	}

}
ptr<cType> sum1D(ptr<cType> a) {
	ptr<cType> result = new cType();
	sum1D(a, result);
	return result;
}
ptr<cType> sum2D(ptr<cType> a) {
	ptr<cType> result = new cType({});

	// pull in reference to save compute
	vector<ptr<cType>>* aVec = &a->vVector;

	for (int i = 0; i < aVec->size(); i++) {
		result->vVector.push_back(sum1D(aVec->at(i)));
	}
	return result;
}

void concat(ptr<cType> a, ptr<cType> b, ptr<cType> output) {

	// pull in reference to save compute
	vector<ptr<cType>>* aVec = &a->vVector;
	vector<ptr<cType>>* bVec = &b->vVector;
	vector<ptr<cType>>* outVec = &output->vVector;

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
void concat(vector<ptr<cType>> vec, ptr<cType> output) {

	// pull in reference to save compute
	vector<ptr<cType>>* outVec = &output->vVector;

	int index = 0;
	for (int i = 0; i < vec.size(); i++) {

		vector<ptr<cType>>* cTypeVec = &vec.at(i)->vVector;
		for (int j = 0; j < cTypeVec->size(); j++) {
			outVec->at(index) = cTypeVec->at(j);
			index++;
		}

	}

}
ptr<cType> concat(ptr<cType> a, ptr<cType> b) {

	vector<ptr<cType>>* aVec = &a->vVector;
	vector<ptr<cType>>* bVec = &b->vVector;

	ptr<cType> result = new cType({});

	for (int i = 0; i < aVec->size(); i++) {
		result->vVector.push_back(new cType());
	}
	for (int i = 0; i < bVec->size(); i++) {
		result->vVector.push_back(new cType());
	}

	concat(a, b, result);
	return result;

}

ptr<vector<int>> randomDist(int count, int incMin, int excMax, bool replace) {

	vector<int> allPossibilities = vector<int>();
	for (int i = incMin; i < excMax; i++) {
		allPossibilities.push_back(i);
	}

	ptr<vector<int>> result = new vector<int>();
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
double actFunc::deriv(double* x, double* y) {
	return 1;
}

double actFuncSm::eval(double x) {
	return 1 / (1 + exp(-x));
}
double actFuncSm::deriv(double* x, double* y) {
	return *y * (1 - *y);
}

double actFuncTh::eval(double x) {
	return tanh(x);
}
double actFuncTh::deriv(double* x, double* y) {
	return 1 / (pow(cosh(*x), 2));
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
double actFuncLR::deriv(double* x, double* y) {
	if (*y > 0) {
		return 1;
	}
	else {
		return m;
	}
}