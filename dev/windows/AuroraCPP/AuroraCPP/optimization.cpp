#pragma once
#include "optimization.h"

#pragma region genePool

genePool::genePool(uniform_real_distribution<double> _mutDis, default_random_engine _randEngine, vector<param*>* _params) {

	this->mutDis = _mutDis;
	this->randEngine = _randEngine;
	this->params = _params;

}
genePool::genePool(uniform_real_distribution<double> _mutDis, default_random_engine _randEngine, vector<param*>* _params, int _genSize, double _mutProb) {

	assert(_genSize > 0);
	assert(_mutProb >= 0 && _mutProb <= 1);
	this->mutDis = _mutDis;
	this->randEngine = _randEngine;
	this->params = _params;
	this->genSize = _genSize;
	this->setMutProb(_mutProb);

}

void genePool::initParent() {

	assert(params != nullptr && params->size() > 0);
	parentStates = make1D(params->size());

}
void genePool::initChildren() {

	assert(genSize > 0);
	assert(params != nullptr);
	assert(params->size() > 0);

	childrenStates = make2D(genSize, params->size());

}

void genePool::populateParams(int childIndex) {

	vector<ptr<cType>>* childrenStatesVec = &childrenStates->vVector;
	ptr<cType> childStates = childrenStatesVec->at(childIndex);
	vector<ptr<cType>>* childStatesVec = &childStates->vVector;

	for (int paramIndex = 0; paramIndex < childStatesVec->size(); paramIndex++) {
		params->at(paramIndex)->state = childStatesVec->at(paramIndex)->vDouble;
	}

}

void genePool::populateParent() {

	vector<ptr<cType>>* parentStatesVec = &parentStates->vVector;

	for (int paramIndex = 0; paramIndex < params->size(); paramIndex++) {

		parentStatesVec->at(paramIndex)->vDouble = params->at(paramIndex)->state;

	}

}
void genePool::birthGeneration() {

	assert(genSize > 0);
	assert(params != nullptr);
	assert(params->size() > 0);
	assert(dividend > 0);

	vector<ptr<cType>>* parentStatesVec = &parentStates->vVector;
	vector<ptr<cType>>* childrenStatesVec = &childrenStates->vVector;

	for (int childIndex = 0; childIndex < genSize; childIndex++) {

		vector<ptr<cType>>* childStatesVec = &childrenStatesVec->at(childIndex)->vVector;

		for (int paramIndex = 0; paramIndex < params->size(); paramIndex++) {
			param* p = params->at(paramIndex);
			ptr<cType> parentParamState = parentStatesVec->at(paramIndex);
			ptr<cType> childParamState = childStatesVec->at(paramIndex);
			if (rand() % dividend <= 1) {
				childParamState->vDouble = parentParamState->vDouble - p->learnRate * mutDis(randEngine);
			}
			else {
				childParamState->vDouble = parentParamState->vDouble;
			}
		}
	}

}

void genePool::makeParent(int childIndex) {

	vector<ptr<cType>>* childrenStatesVec = &childrenStates->vVector;

	copy1D(childrenStatesVec->at(childIndex), parentStates);

}

void genePool::setGenSize(int _genSize) {

	assert(_genSize > 0);
	genSize = _genSize;

}
void genePool::setMutProb(double _mutProb) {

	assert(_mutProb >= 0 && _mutProb <= 1);
	double dDividend = (double)1 / _mutProb;
	dividend = (int)dDividend;

}
void genePool::setParams(vector<param*>* _params) {

	params = _params;

}


#pragma endregion

