#pragma once
#include "aurora.h"
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

	tensor x = {
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	};
	tensor y = {
		{0},
		{1},
		{1},
		{0},
	};

	vector<param_mom*> pl = vector<param_mom*>();

	ptr<sequential> s = pseudo::tnn({ 2, 5, 1 }, pseudo::nlr(0.3), pl);
	s->compile();

	pseudo::init(pl, 10, -1, 1, 0.2, 0.9);

	printf("");

	for (int epoch = 0; epoch < 1000000; epoch++) {
		
		if (epoch % 10000 == 0) {
			printf("\033[%d;%dH", 0, 0);
			std::cout << epoch << std::endl;
		}

		for (int tsIndex = 0; tsIndex < x.size(); tsIndex++) {
			s->cycle(x[tsIndex], y[tsIndex]);
			if (epoch % 10000 == 0)
				std::cout << x[tsIndex].to_string() << " " << s->y.to_string() << std::endl;
		}

		for (param_mom* pmt : pl) {
			pmt->momentum() = pmt->beta() * pmt->momentum() + (1 - pmt->beta()) * pmt->gradient();
			pmt->state() -= pmt->learn_rate() * pmt->momentum();
			pmt->gradient() = 0;
		}

	}

	for (param_mom* pmt : pl) {
		std::cout << pmt->state() << std::endl;
	}

	return 0;
}