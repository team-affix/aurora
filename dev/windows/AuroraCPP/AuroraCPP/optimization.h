#pragma once
#include "superHeader.h"
#include "maths.h"

class param {
public:
	double state;
	double learnRate;
};

class paramSgd : public param {
public:
	double gradient;
};

class paramMom : public paramSgd {
public:
	double momentum;
	double beta;
};

class paramMut : public param {
public:
	double rcv;
	double prevState;
	double beta;
	double momentum;
};

class genePool {
public:
	genePool(uniform_real_distribution<double> _mutDis, default_random_engine _randEngine, vector<param*>* _params);
	genePool(uniform_real_distribution<double> _mutDis, default_random_engine _randEngine, vector<param*>* _params, int _genSize, double _mutProb);

public:
	void initParent();
	void initChildren();

public:
	// populates the paramVec with the child's states
	void populateParams(int childIndex);

public:
	// populates the parent's states with the paramVec
	void populateParent();
	// populates the children's states each with mutated versions of the parent's states
	void birthGeneration();

public:
	// populates the parent's states with the intended child's states
	void makeParent(int childIndex);

public:
	// if the user wants to change the probability of a mutation occuring
	void setGenSize(int _genSize);
	void setMutProb(double _mutProb);
	void setParams(vector<param*>* _params);

private:
	int genSize;
	int dividend;
	vector<param*>* params;
	ptr<cType> parentStates;
	ptr<cType> childrenStates;

	uniform_real_distribution<double> mutDis;
	default_random_engine randEngine;
};