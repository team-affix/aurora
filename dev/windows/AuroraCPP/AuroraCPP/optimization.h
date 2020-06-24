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
	double momentum;
	double prevState;
};