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


void add0D(sPtr<cType> a, sPtr<cType> b, sPtr<cType> output) {
	output->vDouble = a->vDouble + b->vDouble;
}
void add1D(sPtr<cType> a, sPtr<cType> b, sPtr<cType> output) {

	// pull in reference to save compute
	vector<sPtr<cType>>* aVec = &a->vVector;
	vector<sPtr<cType>>* bVec = &b->vVector;
	vector<sPtr<cType>>* outVec = &output->vVector;
	
	// throw exception if vectors are of inequal sizes
	assert(aVec->size() == bVec->size());

	for (int i = 0; i < aVec->size(); i++) {
		outVec->at(i)->vDouble = aVec->at(i)->vDouble + bVec->at(i)->vDouble;
	}

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
	assert(aVec->size() == bVec->size());

	for (int i = 0; i < aVec->size(); i++) {
		outVec->at(i)->vDouble = aVec->at(i)->vDouble - bVec->at(i)->vDouble;
	}

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
	assert(aVec->size() == bVec->size());

	for (int i = 0; i < aVec->size(); i++) {
		outVec->at(i)->vDouble = aVec->at(i)->vDouble * bVec->at(i)->vDouble;
	}

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
	assert(aVec->size() == bVec->size());

	for (int i = 0; i < aVec->size(); i++) {
		outVec->at(i)->vDouble = aVec->at(i)->vDouble / bVec->at(i)->vDouble;
	}

}