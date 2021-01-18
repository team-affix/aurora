#pragma once
#include "data.h"
#include "math.h"
#include "modeling.h"
#include "optimization.h"
#include "pseudo.h"
#include <random>
#include <iostream>

using namespace aurora;
using namespace aurora::data;
using namespace aurora::math;
using namespace aurora::modeling;
using namespace aurora::optimization;
using std::default_random_engine;
using std::uniform_real_distribution;

int main() {

	vector<param_sgd*> pl = vector<param_sgd*>();

	ptr<sequential> s = pseudo::tnn({ 2, 5, 1 }, { pseudo::nlr(0.3), pseudo::nlr(0.3), pseudo::nlr(0.3) }, pl);
	s->compile();

	default_random_engine dre(-26);
	uniform_real_distribution<double> urd(-1, 1);

	for (param_sgd* pmt : pl) {
		pmt->state() = urd(dre);
		pmt->learn_rate() = 0.2;
	}

	tensor x = {
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1}
	};
	tensor y = {
		{0},
		{1},
		{1},
		{0}
	};

	for (int epoch = 0; true; epoch++) {

		if(epoch % 1000 == 0)
			printf("\033[%d;%dH", 0, 0);

		for (int tsIndex = 0; tsIndex < x.size(); tsIndex++) {
			s->cycle(x[tsIndex], y[tsIndex]);
			if (epoch % 1000 == 0)
				std::cout << x[tsIndex].to_string() << " " << s->y.to_string() << std::endl;	
		}

		for (param_sgd* pmt : pl) {
			pmt->state() -= pmt->learn_rate() * pmt->gradient();
			pmt->gradient() = 0;
		}

	}

	return 0;
}