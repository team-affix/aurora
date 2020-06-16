#pragma once
#include "main.h"

using namespace maths;
using namespace modeling;
using namespace optimization;
using namespace sessions;

void main() {

	sequential_bpg neuron_leaky_relu = sequential_bpg();
	neuron_leaky_relu.push_back(new bias_bpg());
	neuron_leaky_relu.push_back(new activate_bpg(new activation_leakyRelu(0.05)));

	sequential_bpg* s = new sequential_bpg();
	s->push_back(new layer_bpg(2, &neuron_leaky_relu));
	s->push_back(new weightJunction_bpg(2, 5));
	s->push_back(new layer_bpg(5, &neuron_leaky_relu));
	s->push_back(new weightJunction_bpg(5, 1));
	s->push_back(new layer_bpg(1, &neuron_leaky_relu));

	vector<parameter**> parameterPtrs = vector<parameter**>();
	vector<parameter_sgd*> parameters = vector<parameter_sgd*>();
	s->foreach([&parameterPtrs](model* m) {
		initialize_model(m, &parameterPtrs);
	});

	std::uniform_real_distribution<double> unif(-1, 1);
	std::default_random_engine re(6);
	for (auto ptr : parameterPtrs) {
		parameter_sgd* p = new parameter_sgd();
		p->learnRate = 0.0002;
		p->state = unif(re);
		p->gradient = 0;
		*ptr = p;
		parameters.push_back(p);
	}

	carryType inputs = {
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1}
	};
	carryType desired = {
		{0},
		{0},
		{1},
		{1}
	};

	int epoch = 0;
	tens1D dist = { 0, 1, 2, 3 };

	s->yGrad = shared_ptr<carryType>(new carryType(vector<shared_ptr<carryType>>()));
	s->yGrad->value_vector.push_back(shared_ptr<carryType>(new carryType(0)));
	while (epoch < 10000000) {
		tens1D order = tens1D(4, dist, false);
		tens1D signals = tens1D(1);
		for (int i = 0; i < inputs.value_vector.size(); i++) {
			int tsIndex = order.at(i);
			s->x = inputs.value_vector[tsIndex];
			s->fwd();
			zero(&s->yGrad->value_vector);
			subtract(&s->y->value_vector, &desired.value_vector.at(tsIndex)->value_vector, &s->yGrad->value_vector);
			signals = signals + ((tens1D)s->yGrad->value_vector).abs();
			s->bwd();
		}
		for (int i = 0; i < parameters.size(); i++) {
			parameter_sgd* p = parameters.at(i);
			p->state -= p->learnRate * p->gradient;
			p->gradient = 0;
		}
		if (epoch % 1000 == 0) {
			cout << signals.getSum() << endl;
		}

		epoch++;
	}
}