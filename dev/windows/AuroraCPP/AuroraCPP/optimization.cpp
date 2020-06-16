#pragma once
#include "optimization.h"
using namespace maths;
using namespace optimization;

#pragma region optimizer_mutation
void optimizer_mutation::update_state(parameter_mutation* p, int domainSize) {
	double x = (rand() % domainSize) - 0.5 * domainSize;
	p->rcv = p->learnRate * pow(x, 3);
	p->state -= range(p->learnRate / (p->rcv - p->momentum), -p->learnRate, p->learnRate);
}
void optimizer_mutation::keep_state(parameter_mutation* p) {
	p->momentum = (p->beta * p->momentum) + (1 - p->beta) * p->rcv;
	p->state_previous = p->state;
}
void optimizer_mutation::roll_back_state(parameter_mutation* p) {
	p->state = p->state_previous;
}
#pragma endregion
#pragma region optimizer_sgd
void optimizer_sgd::update_state(parameter_sgd* p) {
	p->state -= p->learnRate * p->gradient;
}
void optimizer_sgd::clear_gradient(parameter_sgd* p) {
	p->gradient = 0;
}
#pragma endregion
#pragma region optimizer_momentum
void optimizer_momentum::update_state(parameter_momentum* p) {
	p->state -= p->momentum;
}
void optimizer_momentum::update_momentum(parameter_momentum* p) {
	p->momentum = (p->beta * p->momentum) + (1 - p->beta) * p->gradient;
}
#pragma endregion
