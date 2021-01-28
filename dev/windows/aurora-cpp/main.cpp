#pragma once
#include "aurora.h"
#include <random>
#include <iostream>
#include <assert.h>
#include <thread>
#include <mutex>
#include <fstream>
#include <time.h>

using namespace aurora;
using namespace aurora::data;
using namespace aurora::math;
using namespace aurora::modeling;
using namespace aurora::optimization;
using std::default_random_engine;
using std::uniform_real_distribution;

std::string pl_export(vector<param_sgd*>& a_pl) {
	std::string result;
	for (int i = 0; i < a_pl.size(); i++)
		result += std::to_string(a_pl[i]->state()) + "\n";
	result.pop_back();
	return result;
}

int num_lines(std::string file_name) {
	int count = 0;
	string line;
	/* Creating input filestream */
	std::ifstream ifs(file_name);
	while (std::getline(ifs, line))
		count++;
	return count;
}

void tensor_test() {

	tensor vec_1 = { 0, 1, 2 };
	tensor vec_2 = vec_1.link();
	vec_2[0].val() = 10;
	assert(vec_1[0].val() == 10);

	vec_2.set({ 1, 2, 3 });
	assert(vec_1[0].val() == 1);
	assert(vec_1[1].val() == 2);
	assert(vec_1[2].val() == 3);

	tensor mat_1 = tensor::new_2d(10, 10);
	tensor mat_1_unroll = mat_1.unroll();
	assert(mat_1_unroll.size() == 100);

	mat_1_unroll.set(tensor::new_1d(100, 10));
	assert(mat_1[0][0] == 10);

	tensor mat_2 = mat_1_unroll.roll(10);
	assert(mat_2.size() == 10);


}

void tnn_xor_test() {

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

	pseudo::pl_init(pl, 10, -1, 1, 0.2, 0.9);

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
}

void tnn_test() {
	tensor x = {
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
		{1.1, 1},
		{1.2, 1},
		{1, 1.1},
		{1, 1.2},
		{1, 1.3},
		{1, 1.4},
		{1, 1.5},
		{1.6, 1},
		{1.7, 1},
		{1.8, 1},
		{1.75, 1},
		{1.85, 1},
	};

	tensor y = {
		{0},
		{1},
		{1},
		{0},
		{3},
		{4},
		{5},
		{9},
		{10},
		{5.5},
		{3.2},
		{7.8},
		{13},
		{14.5},
		{19},
		{20},
	};

	vector<param_mom*> pl = vector<param_mom*>();

	ptr<sequential> s = pseudo::tnn({ 2, 17, 1 }, pseudo::nlr(0.3), pl);

	pseudo::pl_init(pl, 10, -1, 1, 0.2, 0.9);

	printf("");

	for (int epoch = 0; epoch < 1000000; epoch++) {

		if (epoch % 10000 == 0) {
			printf("\033[%d;%dH", 0, 0);
			std::cout << epoch << std::endl;
		}

		for (int tsIndex = 0; tsIndex < x.size(); tsIndex++)
			s->cycle(x[tsIndex], y[tsIndex]);

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

void tnn_multithread_test() {

	tensor x = {
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
		{1.1, 1},
		{1.2, 1},
		{1, 1.1},
		{1, 1.2},
		{1, 1.3},
		{1, 1.4},
		{1, 1.5},
		{1.6, 1},
		{1.7, 1},
		{1.8, 1},
		{1.75, 1},
		{1.85, 1},
	};

	tensor y = {
		{0},
		{1},
		{1},
		{0},
		{3},
		{4},
		{5},
		{9},
		{10},
		{5.5},
		{3.2},
		{7.8},
		{13},
		{14.5},
		{19},
		{20},
	};

	vector<param_mom*> pl = vector<param_mom*>();

	const int numClones = 16;

	vector<ptr<sequential>> clones = vector<ptr<sequential>>(numClones);
	clones[0] = pseudo::tnn({ 2, 17, 1 }, pseudo::nlr(0.3), pl);

	for (int i = 1; i < clones.size(); i++)
		clones[i] = (sequential*)clones.front()->clone();

	for (int i = 0; i < clones.size(); i++)
		clones[i]->compile();

	pseudo::persistent_thread threads[numClones];

	pseudo::pl_init(pl, 25, -1, 1, 0.0002, 0.9);

	printf("");

	for (int epoch = 0; true; epoch++) {

		if (epoch % 10000 == 0) {
			printf("\033[%d;%dH", 0, 0);
			std::cout << epoch << std::endl;
		}

		for (int tsIndex = 0; tsIndex < numClones; tsIndex++) {
			ptr<sequential>& seq = clones[tsIndex];
			tensor& seq_x = x[tsIndex];
			tensor& seq_y = y[tsIndex];
			threads[tsIndex].execute([&] {
				seq->cycle(seq_x, seq_y);
			});
		}

		for (pseudo::persistent_thread& thd : threads) {
			thd.join();
		}

		if(epoch % 10000 == 0)
			for (int tsIndex = 0; tsIndex < numClones; tsIndex++)
				std::cout << x[tsIndex].to_string() << " " << clones[tsIndex]->y.to_string() << std::endl;

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

void sync_xor_test() {

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

	vector<param_sgd*> pl = vector<param_sgd*>();
	ptr<sync> s_prev = new sync(pseudo::tnn({ 2, 5, 1 }, pseudo::nlr(0.3), pl));
	ptr<sync> s = (sync*)s_prev->clone();
	pseudo::pl_init(pl, 25, -1, 1, 0.2);

	s->prep(4);
	s->compile();

	s->unroll(4);

	for (int epoch = 0; epoch < 1000000; epoch++) {
		s->cycle(x, y);

		if (epoch % 10000 == 0)
			std::cout << x.to_string() << std::endl << s->y.to_string() << std::endl << std::endl;

		for (param_sgd* pmt : pl) {
			pmt->state() -= pmt->learn_rate() * pmt->gradient();
			pmt->gradient() = 0;
		}
	}

}

int random(int min, int max) {
	max -= 1;
	return min + (rand() % static_cast<int>(max - min + 1));
}

tensor get_encoder_x(size_t order_1_len) {
	tensor result = tensor::new_1d(order_1_len);
	for (int i = 0; i < order_1_len; i++)
		result[i].val() = random(0, 256);
	return result;
}

tensor get_encoder_x(size_t order_2_len, size_t order_1_len) {
	tensor result = tensor::new_2d(order_2_len, order_1_len);
	for (int i = 0; i < order_2_len; i++)
		result[i] = get_encoder_x(order_1_len);
	return result;
}

tensor get_encoder_x(size_t order_3_len, size_t order_2_len, size_t order_1_len) {
	tensor result = tensor::new_1d(order_3_len);
	for (int i = 0; i < order_3_len; i++)
		result[i] = get_encoder_x(order_2_len, order_1_len);
	return result;
}

template<class T>
bool true_for(std::vector<T> vec, std::function<bool(T)> func) {
	for (int i = 0; i < vec.size(); i++)
		if (!func(vec[i])) return false;
	return true;
}

void encoder_test() {

	const int encoded_len = 32;
	const int h_3_len = encoded_len * 2;
	const int h_2_len = encoded_len * 4;
	const int h_1_len = encoded_len * 6;
	const int x_order_1_len = encoded_len * 8 + 1;
	const int x_order_2_len = 3000;
	const int mini_batch_len = 100;
	const std::string file_name = "ae0";
	const bool import_from_file = true;

	std::cout << "GATHERING TRAINING DATA" << std::endl;
	tensor train_x = get_encoder_x(x_order_2_len, x_order_1_len);

	vector<param_sgd*> pl = vector<param_sgd*>();
	std::cout << "INSTANTIATING MODEL" << std::endl;
	ptr<sequential> s = pseudo::tnn({ x_order_1_len, h_1_len, h_2_len, h_3_len, encoded_len, h_3_len, h_2_len, h_1_len, x_order_1_len }, pseudo::nlr(0.3), pl);

	uniform_real_distribution<double> urd((double)-1 / pl.size(), (double)1 / pl.size());
	default_random_engine dre(801);

	if (import_from_file && (num_lines(file_name) == pl.size())) {
		std::ifstream ifs(file_name);
		for (int i = 0; i < pl.size(); i++) {
			std::string str;
			std::getline(ifs, str);
			pl[i]->state() = std::stod(str);
		}
	}
	else {
		std::cout << "INITIALIZING PARAMETER STATES" << std::endl;
		for (param_sgd* pmt : pl)
			pmt->state() = urd(dre);
	}

	for (param_sgd* pmt : pl) {
		pmt->learn_rate() = 0.000000002;
	}

	std::cout << "COMPILING MODEL" << std::endl;
	s->compile();

	double best_cost = INFINITY;

	const int checkpoint_interval = 5;
	const int max_fail_lr_update = 35;

	int fail_index = 0;

	double checkpoint_cost = 0;

	for (int epoch = 0; true; epoch++) {
		double epoch_cost = 0;
		for (int i = 0; i < mini_batch_len; i++) {
			int ts_index = random(0, x_order_2_len);
			s->cycle(train_x[ts_index], train_x[ts_index]);
			if(epoch % checkpoint_interval == 0)
				epoch_cost += s->y_grad.abs_1d().sum_1d().val();
		}

		checkpoint_cost += epoch_cost;

		if (epoch % checkpoint_interval == 0 && epoch != 0) {
			std::cout << checkpoint_cost << " [" << checkpoint_cost / x_order_1_len / mini_batch_len / checkpoint_interval << "]" << std::endl;
			if (true_for<param_sgd*>(pl, [&](param_sgd* pmt) {
				return !isnan(pmt->state());
			}) && checkpoint_cost < best_cost) {
				best_cost = checkpoint_cost;
				fail_index = 0;
				std::ofstream ofs(file_name, std::ofstream::out | std::ofstream::trunc);
				ofs << pl_export(pl) << std::endl;
				ofs.close();
			} else {
				std::cout << "----^^^^^----" << std::endl;
				fail_index++;
				if (fail_index >= max_fail_lr_update) {
					for (param_sgd* pmt : pl) {
						pmt->learn_rate() *= 0.1;
					}
					fail_index = 0;
				}
			}
			checkpoint_cost = 0;
		}
			
		for (param_sgd* pmt : pl) {
			int rcv = 1;
			/*if (random(0, 20) == 0)
				rcv = 30;*/
			pmt->state() -= rcv * pmt->learn_rate() * pmt->gradient();
			pmt->gradient() = 0;
		}
	}

}

int main() {

	srand(time(NULL));

	encoder_test();

	return 0;
}