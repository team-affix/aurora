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

void tnn_test() {

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

	default_random_engine dre(-26);
	uniform_real_distribution<double> urd(-1, 1);

	for (param_mom* pmt : pl) {
		pmt->state() = urd(dre);
		pmt->learn_rate() = 0.02;
		pmt->beta() = 0.9;
	}

	printf("");

	for (int epoch = 0; epoch < 1000000; epoch++) {

		if (epoch % 10000 == 0)
			printf("\033[%d;%dH", 0, 0);

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

}

void tensor_test() {
	tensor t1 = { 0, 1, 2, 3 };
	tensor t2 = t1.link();
	t2.resize(10);
}

int main() {
	tnn_test();
	return 0;
}