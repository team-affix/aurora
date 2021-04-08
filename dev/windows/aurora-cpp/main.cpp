#pragma once
#include "aurora.h"
#include <random>
#include <iostream>
#include <assert.h>
#include <thread>
#include <mutex>
#include <fstream>
#include <time.h>
#include <filesystem>
#include <windows.h>
#include <fstream>

using namespace aurora;
using namespace aurora::data;
using namespace aurora::math;
using namespace aurora::models;
using namespace aurora::params;
using namespace aurora::evolution;
using std::default_random_engine;
using std::uniform_real_distribution;
using std::uniform_int_distribution;
using std::ifstream;
using std::ofstream;

std::string pl_export(vector<param_sgd*>& a_pl) {
	std::string result;
	for (int i = 0; i < a_pl.size(); i++)
		result += std::to_string(a_pl[i]->state()) + "\n";
	result.pop_back();
	return result;
}

template<class T>
void pl_export_to_file(std::string file_name, vector<T*> a_pl) {
	for(int i = 0; i < a_pl.size(); i++)
		if (isnan(a_pl[i]->state()) || isinf(a_pl[i]->state())) {
			std::cout << "ERROR: PARAM IS NAN OR INF" << std::endl;
			return;
		}
	ofstream ofs(file_name);
	for (int i = 0; i < a_pl.size(); i++)
		ofs << a_pl[i]->state() << std::endl;
	ofs.close();
}

template<class T>
void pl_import_from_file(std::string file_name, vector<T*> a_pl) {
	ifstream ifs(file_name);
	double state = 0;
	for (int i = 0; ifs >> state; i++)
		a_pl[i]->state() = state;
	ifs.close();
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

	tensor mat_3 = tensor::new_2d(10, 10);
	tensor mat_4;

	{
		// mat_5 WILL ONLY EXIST IN THIS SCOPE, THEN WILL BE DISCARDED
		tensor mat_5 = mat_3.range_2d(0, 0, 2, 2);
		mat_4.group_join(mat_5);
	}
	mat_4.pop(tensor::new_2d(2, 2, 1));
	assert(mat_3[0][0] == 1);
	assert(mat_3[0][1] == 1);
	assert(mat_3[1][0] == 1);
	assert(mat_3[1][1] == 1);

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

	uniform_real_distribution<double> urd(-1, 1);
	default_random_engine re(25);

	ptr<sequential> s = pseudo::tnn({ 2, 5, 1 }, pseudo::nlr(0.3), [&](ptr<param>& pmt) {
		pmt = new param_mom(urd(re), 0.02, 0, 0, 0.9);
		pl.push_back((param_mom*)pmt.get());
	});

	s->compile();

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

		for (param_mom* pmt : pl)
			pmt->update();
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

	ptr<sequential> s = pseudo::tnn({ 2, 17, 1 }, pseudo::nlr(0.3), pseudo::dump_pmt(pl));

	double state_structure = 1 / (double)pl.size();
	uniform_real_distribution<double> urd(-state_structure, state_structure);

	s->pmt_wise(pseudo::init_pmt(urd, 0.02));

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

	vector<param_sgd_mt*> pl = vector<param_sgd_mt*>();

	uniform_real_distribution<double> urd(-1, 1);
	default_random_engine re(26);

	const int numClones = 16;

	sync s = sync(numClones, pseudo::tnn({ 2, 25, 1 }, pseudo::nlr(0.3), [&](ptr<param>& pmt) {
		pmt = new param_sgd_mt(urd(re), 0.0002, 0);
		pl.push_back((param_sgd_mt*)pmt.get());
	}));

	s.compile();
	s.unroll(numClones);
	
	pseudo::persistent_thread threads[numClones];

	printf("");

	for (int epoch = 0; true; epoch++) {

		if (epoch % 10000 == 0) {
			printf("\033[%d;%dH", 0, 0);
			std::cout << epoch << std::endl;
		}

		for (int tsIndex = 0; tsIndex < numClones; tsIndex++) {
			sequential& seq = *(sequential*)s.unrolled[tsIndex].get();
			tensor& seq_x = x[tsIndex];
			tensor& seq_y = y[tsIndex];
			threads[tsIndex].execute([&] {
				seq.cycle(seq_x, seq_y);
			});
		}

		for (pseudo::persistent_thread& thd : threads) {
			thd.join();
		}

		if (epoch % 10000 == 0)
			std::cout << s.y.to_string() << std::endl;

		for (param_sgd_mt* pmt : pl) {
			pmt->state() -= pmt->learn_rate() * pmt->gradient();
			pmt->gradient() = 0;
		}

	}

	for (param_sgd* pmt : pl) {
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

	uniform_real_distribution<double> urd(-1, 1);
	default_random_engine re(25);

	vector<param_sgd*> pl = vector<param_sgd*>();
	ptr<sync> s_prev = new sync(4, pseudo::tnn({ 2, 5, 1 }, pseudo::nlr(0.3), [&](ptr<param>& pmt) {
		pmt = new param_sgd(urd(re), 0.02, 0);
		pl.push_back((param_sgd*)pmt.get());
	}));
	ptr<sync> s = (sync*)s_prev->clone();
	
	s->compile();

	s->unroll(4);

	for (int epoch = 0; epoch < 1000000; epoch++) {
		s->cycle(x, y);

		if (epoch % 10000 == 0) {
			printf("\033[%d;%dH", 0, 0);
			std::cout << epoch << std::endl;
		}

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

double random_d(double min, double max, default_random_engine& re) {
	uniform_real_distribution<double> urd(min, max);
	return urd(re);
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
	ptr<sequential> s = pseudo::tnn({ x_order_1_len, h_1_len, h_2_len, h_3_len, encoded_len, h_3_len, h_2_len, h_1_len, x_order_1_len }, pseudo::nlr(0.3), [&](ptr<param>& pmt) {
		pmt = new param_sgd();
		pl.push_back((param_sgd*)pmt.get());
	});

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
		for (int i = 0; i < pl.size(); i++)
			pl[i]->state() = urd(dre);
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

void lstm_test() {
	tensor x0 = {
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	};
	tensor x1 = {
		{0, 1},
		{0, 1},
		{1, 0},
		{1, 1},
	};

	tensor y0 = {
		{0},
		{1},
		{1},
		{0},
	};
	tensor y1 = {
		{1},
		{1},
		{1},
		{1},
	};

	uniform_real_distribution<double> urd(-1, 1);
	default_random_engine re(25);

	const int lstm_units = 5;

	vector<param_sgd*> pv;

	auto pmt_init = [&](ptr<param>& pmt) {
		pmt = new param_sgd(urd(re), 0.02, 0);
		pv.push_back((param_sgd*)pmt.get());
	};

	sync* s0 = new sync(4, pseudo::tnn({ 2, lstm_units }, pseudo::nlr(0.3), pmt_init));
	lstm* l = new lstm(lstm_units, 4, pmt_init);
	sync* s1 = new sync(4, pseudo::tnn({ lstm_units, 1 }, pseudo::nlr(0.3), pmt_init));

	ptr<sync> s0_ptr = s0;
	ptr<lstm> l0_ptr = l;
	ptr<sync> s1_ptr = s1;

	sequential s = sequential({ s0, l, s1 });

	s.compile();

	s0->unroll(4);
	l->unroll(4);
	s1->unroll(4);

	const int checkpoint_interval = 10000;

	for (int epoch = 0; true; epoch++) {
		s.cycle(x0, y0);
		if (epoch % checkpoint_interval == 0)
			std::cout << "S0: " << s.y.to_string() << std::endl;
		s.cycle(x1, y1);
		if (epoch % checkpoint_interval == 0)
			std::cout << "S1: " << s.y.to_string() << std::endl;
		for (param_sgd* pmt : pv) {
			pmt->state() -= pmt->learn_rate() * pmt->gradient();
			pmt->gradient() = 0;
		}
	}
}

void input_gd() {

#pragma region HYPER CONFIGURE
	const double learn_rate = 0.002;
	const double beta = 0.9;
#pragma endregion
#pragma region RANDOM INSTANTIATE
	uniform_real_distribution<double> urd(-1, 1);
	default_random_engine re(25);
#pragma endregion
#pragma region ORDER 0 INSTANTIATE
	tensor x_0 = {
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	};
	tensor y_0 = {
		{0},
		{1},
		{1},
		{0},
	};

	vector<param_mom*> pv_0;
	ptr<sync> s_0 = new sync(x_0.size(), pseudo::tnn({ 2, 5, 1 }, pseudo::nlr(0.3), [&](ptr<param>& pmt) {
		pmt = new param_mom(urd(re), learn_rate, 0, 0, beta);
		pv_0.push_back((param_mom*)pmt.get());
	}));
#pragma endregion
#pragma region ORDER 1 INSTANTIATE
	const int x_1_height = 20;
	tensor x_1 = tensor::new_2d(x_1_height, pv_0.size(), urd, re);
	vector<param_mom*> pv_1;
	ptr<sequential> s_1 = pseudo::tnn({ pv_0.size(), pv_0.size() * 2, 1 }, pseudo::nlr(0.3), [&](ptr<param>& pmt) {
		pmt = new param_mom(urd(re), learn_rate, 0, 0, beta);
		pv_1.push_back((param_mom*)pmt.get());
	});
#pragma endregion
#pragma region TENSOR INSTANTIATE
	tensor pv_0_states = tensor::new_1d(pv_0.size());
	for (int i = 0; i < pv_0.size(); i++)
		pv_0_states[i].val_ptr.link(pv_0[i]->state_ptr);
	tensor deviant_input = tensor::new_1d(pv_0.size(), urd, re);
	tensor deviant_output = { 0 };
	tensor deviant_learn_rate_tensor = tensor::new_1d(pv_0.size(), learn_rate);
	tensor deviant_momentum_tensor = tensor::new_1d(pv_0.size(), 0);
	tensor deviant_beta_tensor = tensor::new_1d(pv_0.size(), beta);
	tensor deviant_beta_compliment_tensor = tensor::new_1d(pv_0.size(), 1 - beta);
#pragma endregion
#pragma region COMPILE
	s_0->compile();
	s_0->unroll(x_0.size());
	s_1->compile();
#pragma endregion
#pragma region CONFIGURE TRAINING
	const double min_ts_cost = 1E-1;
	const int checkpoint_interval = 100;
	const int deviant_update_amount = 10;
#pragma endregion
#pragma region GET ORDER 1 TRAINING SET Y
	tensor y_1 = tensor::new_2d(x_1.size(), 1);
	auto get_tsy_1 = [&](tensor tsx_1) {
		pv_0_states.pop(tsx_1);
		s_0->fwd(x_0);
		s_0->signal(y_0);
		return s_0->y_grad.abs_2d().sum_2d();
	};
	for (int tsIndex = 0; tsIndex < x_1.size(); tsIndex++) {
		y_1[tsIndex] = get_tsy_1(x_1[tsIndex]);
	}
#pragma endregion

	auto update_params = [&](vector<param_mom*> pv) {
		for (param_mom* pmt : pv) {
			pmt->momentum() = pmt->beta() * pmt->momentum() + (1 - pmt->beta()) * pmt->gradient();
			pmt->state() -= pmt->learn_rate() * pmt->gradient();
			pmt->gradient() = 0;
		}
	};

	for (int epoch = 0; true; epoch++) {
		double epoch_cost = 0;
		for (int tsIndex = 0; tsIndex < x_1.size(); tsIndex++) {
			s_1->cycle(x_1[tsIndex], y_1[tsIndex]);
			epoch_cost += s_1->y_grad.abs_1d().sum_1d().val() / x_1.size();
		}

		if (epoch % checkpoint_interval == 0)
			std::cout << epoch_cost << std::endl;

		bool cost_low = epoch_cost < min_ts_cost;

		if (cost_low && epoch % checkpoint_interval == 0) {
			std::cout << "NEW INPUT" << std::endl;
			// GET NEW ORDER 1 TRAINING SET
			for (int deviant_update = 0; deviant_update < deviant_update_amount; deviant_update++) {
				s_1->cycle(deviant_input, deviant_output);
				deviant_momentum_tensor = deviant_momentum_tensor.mul_1d(deviant_beta_tensor).add_1d(deviant_beta_compliment_tensor.mul_1d(s_1->x_grad));
				deviant_input = deviant_input.sub_1d(deviant_momentum_tensor.mul_1d(deviant_learn_rate_tensor));
			}

			// POP BACK OF ORDER 1 TS VECTORS
			if (x_1.size() >= 25) {
				x_1.vec().pop_back();
				y_1.vec().pop_back();
			}

			// APPEND NEW TRAINING SET
			x_1.vec().push_back(deviant_input.clone());
			y_1.vec().push_back(get_tsy_1(deviant_input));

		}

		if (epoch % checkpoint_interval == 0 && cost_low)
			std::cout << "TEST COST: " << get_tsy_1(deviant_input).sum_1d().val() << std::endl;

		update_params(pv_1);
	}
}

void loss_map() {

#pragma region HYPER CONFIGURE
	const double learn_rate = 0.002;
	const double beta = 0.9;
#pragma endregion
#pragma region RANDOM INSTANTIATE
	uniform_real_distribution<double> urd(-1, 1);
	default_random_engine re(25);
#pragma endregion
#pragma region ORDER 0 INSTANTIATE
	tensor x_0 = {
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	};
	tensor y_0 = {
		{0},
		{1},
		{1},
		{0},
	};

	vector<param_mom*> pv_0;
	ptr<sync> s_0 = new sync(x_0.size(), pseudo::tnn({ 2, 5, 1 }, pseudo::nlr(0.3), [&](ptr<param>& pmt) {
		pmt = new param_mom(urd(re), learn_rate, 0, 0, beta);
		pv_0.push_back((param_mom*)pmt.get());
		}));
#pragma endregion
#pragma region ORDER 1 INSTANTIATE
	const int x_1_height = 50;
	tensor x_1 = tensor::new_2d(x_1_height, 1, urd, re);
	vector<param_mom*> pv_1;
	ptr<sequential> s_1 = pseudo::tnn({ 1, pv_0.size(), pv_0.size() }, pseudo::nlr(0.3), [&](ptr<param>& pmt) {
		pmt = new param_mom(urd(re), learn_rate, 0, 0, beta);
		pv_1.push_back((param_mom*)pmt.get());
	});
#pragma endregion
#pragma region TENSOR INSTANTIATE
	tensor pv_0_states = tensor::new_1d(pv_0.size());
	for (int i = 0; i < pv_0.size(); i++)
		pv_0_states[i].val_ptr.link(pv_0[i]->state_ptr);
	tensor deviant_x = { 0 };
#pragma endregion
#pragma region COMPILE
	s_0->compile();
	s_0->unroll(x_0.size());
	s_1->compile();
#pragma endregion
#pragma region CONFIGURE TRAINING
	const double min_ts_cost = 8;
	const int checkpoint_interval = 1000;
	const tensor deviant_update_amount = { 0.1 };
#pragma endregion
#pragma region GET ORDER 1 TRAINING SET Y
	tensor y_1 = tensor::new_2d(x_1.size(), pv_0.size(), urd, re);
	auto get_tsx_1 = [&](tensor tsy_1) {
		pv_0_states.pop(tsy_1);
		s_0->fwd(x_0);
		s_0->signal(y_0);
		return s_0->y_grad.abs_2d().sum_2d();
	};
	for (int tsIndex = 0; tsIndex < x_1.size(); tsIndex++)
		x_1[tsIndex] = get_tsx_1(y_1[tsIndex]);
#pragma endregion

	auto update_params = [&](vector<param_mom*> pv) {
		for (param_mom* pmt : pv)
			pmt->update();
	};

	double prev_test_cost = 10;

	for (int epoch = 0; true; epoch++) {
		double epoch_cost = 0;
		for (int tsIndex = 0; tsIndex < x_1.size(); tsIndex++) {
			s_1->cycle(x_1[tsIndex], y_1[tsIndex]);
			epoch_cost += s_1->y_grad.abs_1d().sum_1d().val() / x_1.size();
		}

		if (epoch % checkpoint_interval == 0)
			std::cout << epoch_cost << std::endl;

		bool cost_low = epoch_cost < min_ts_cost;

		if (cost_low && epoch % checkpoint_interval == 0) {
			std::cout << "NEW INPUT" << std::endl;
			// GET NEW ORDER 1 TRAINING SET
			deviant_x.pop(tensor({ prev_test_cost }).sub_1d(deviant_update_amount));
			s_1->fwd(deviant_x);

			// POP BACK OF ORDER 1 TS VECTORS
			x_1.vec().erase(x_1.vec().begin());
			y_1.vec().erase(y_1.vec().begin());

			// APPEND NEW TRAINING SET
			x_1.vec().push_back(get_tsx_1(s_1->y));
			y_1.vec().push_back(s_1->y);

		}

		if (epoch % checkpoint_interval == 0 && cost_low) {
			prev_test_cost = get_tsx_1(s_1->fwd(deviant_x)).abs_1d().sum_1d().val();
			std::cout << "TEST COST: " << prev_test_cost << std::endl;

		}

		update_params(pv_1);
	}
}

void zil_trader() {

	uniform_real_distribution<double> pmt_urd(-0.1, 0.1);
	uniform_real_distribution<double> zil_urd(0.13, 0.20);
	uniform_int_distribution<int> trade_uid(0, 1);
	default_random_engine re(25);

#pragma region DATA GATHER
	const size_t episode_size = 200;
	tensor value_tensor_2d = tensor::new_2d(1, episode_size, zil_urd, re);
	tensor value_tensor_3d = tensor::new_1d(value_tensor_2d.size());
	for (int i = 0; i < value_tensor_3d.size(); i++)
		value_tensor_3d[i] = value_tensor_2d[i].roll(1);
#pragma endregion
#pragma region PARAMS
	vector<param*> pv;
	auto pmt_init = [&](ptr<param>& pmt) {
		pmt = new param(pmt_urd(re));
		pv.push_back((param*)pmt.get());
	};
#pragma endregion
#pragma region MODEL

	const size_t lstm_units = 20;
	ptr<sync> sync_in = new sync(episode_size, pseudo::tnn({ 1, lstm_units }, pseudo::nlr(0.3), pmt_init));
	ptr<lstm> lstm_mid = new lstm(lstm_units, episode_size, pmt_init);
	ptr<sync> sync_out = new sync(episode_size, pseudo::tnn({ lstm_units, 2 }, { pseudo::nlr(0.3), pseudo::nsm() }, pmt_init));
	sequential s = { sync_in.get(), lstm_mid.get(), sync_out.get() };

	s.compile();
	sync_in->unroll(episode_size);
	lstm_mid->unroll(episode_size);
	sync_out->unroll(episode_size);

#pragma endregion
#pragma region FUNCS
	auto get_actions = [&](tensor& a_action_tensor) {
		vector<bool> result = vector<bool>(a_action_tensor.size());
		for (int i = 0; i < a_action_tensor.size(); i++)
			result[i] = a_action_tensor[i] == 1;
		return result;
	};
	auto get_usd = [&](tensor& a_zil_price_1d, vector<bool>& a_actions) {
		const double start_usd = 100;
		double usd = start_usd;
		double zil = 0;
		bool buy = true;
		for (int i = 0; i < a_actions.size(); i++) {
			if (a_actions[i]) {
				if (buy) {
					zil = usd / a_zil_price_1d[i];
				}
				else {
					usd = zil * a_zil_price_1d[i];
				}
				buy = !buy;
			}
		}
		return usd - start_usd;
	};
	auto get_rewards = [&](genome& a_genome) {
		double result = 0;
		for (int i = 0; i < value_tensor_3d.size(); i++) {
			vector<bool> actions = get_actions(a_genome);
			result += get_usd(value_tensor_2d[i], actions);
		}
		return result;
	};
#pragma endregion
#pragma region MUTATE
	auto mutate = [&](double val) {
		if (rand() % 100 == 0)
			return (double)trade_uid(re);
		return val;
	};
#pragma endregion
#pragma region EVOLVE
	vector<genome> parents = genome(tensor::new_1d(episode_size), mutate).mutate(3);
	for (int epoch = 0; true; epoch++) {
		generation gen1 = generation(genome::mutate(genome::merge(parents, 40)), get_rewards);
		parents = gen1.best(3);
		if (epoch % 10 == 0)
			std::cout << get_rewards(parents[0]) << std::endl;
	}
#pragma endregion

}

std::string& ltrim(std::string& str, const std::string& chars = "\t\n\v\f\r ")
{
	str.erase(0, str.find_first_not_of(chars));
	return str;
}

std::string& rtrim(std::string& str, const std::string& chars = "\t\n\v\f\r ")
{
	str.erase(str.find_last_not_of(chars) + 1);
	return str;
}

std::string& trim(std::string& str, const std::string& chars = "\t\n\v\f\r ")
{
	return ltrim(rtrim(str, chars), chars);
}

tensor get_zil_data() {
	string zil_csv_directory = "D:\\files\\files\\csv\\zil";
	tensor raw;
	for (const auto& entry : std::filesystem::directory_iterator(zil_csv_directory)) {
		tensor csv = tensor::new_2d(num_lines(entry.path().u8string()), 2);
		string line;
		std::ifstream ifs(entry);
		int row = 0;
		while (std::getline(ifs, line)) {
			std::istringstream s(line);
			std::string field;
			int col = 0;
			while (getline(s, field, ',')) {
				string trimmed = trim(field, "\"");
				if (col == 1)
					csv[row][col] = std::stod(trimmed);
				else
					csv[row][col] = std::stod(trimmed) / 1000 / 60;
				col++;
			}
			row++;
		}
		raw.vec().push_back(csv);
	}
	return raw;
}

void zil_mapper() {
	tensor raw = get_zil_data();
	tensor x;
	tensor y;
	const int num_from_each_csv = 10000;
	for (tensor& csv : raw.vec()) {
		for (int ts = 0; ts < num_from_each_csv; ts++)
		{
			int minutes_wait = rand() % 240;

		}
	}
}

struct composite_function {
	~composite_function(){
		for (double i = 0; i < layers.size(); i++)
			delete layers[i];
	}
	vector<function<double(double)>*> layers;
	double operator()(double x) {
		return layers.back()->operator()(x);
	}
};

composite_function get_composite_fn(size_t order, double min_opd, double max_opd, default_random_engine& re) {
	vector<function<double(double)>*> res_vec;
	res_vec.push_back(new function<double(double)>([&](double x) { return x; }));
	for (double i = 1; i < order; i++) {
		function<double(double)>* layer;
		function<double(double)>& previous = *res_vec[i - 1];
		switch (rand() % 4) {
		case 0:
			layer = new function<double(double)>([&](double x) {
				return previous(x) + random_d(min_opd, max_opd, re);
			});
			break;
		case 1:
			layer = new function<double(double)>([&](double x) {
				return previous(x) - random_d(min_opd, max_opd, re);
			});
			break;
		case 2:
			layer = new function<double(double)>([&](double x) {
				return previous(x) * random_d(min_opd, max_opd, re);
			});
			break;
		case 3:
			layer = new function<double(double)>([&](double x) {
				return previous(x) / random_d(min_opd, max_opd, re);
			});
			break;
		}
		res_vec.push_back(layer);
	}
	return { res_vec };
}

void task_encoder() {

	const size_t order_2_set_len = 100;
	const size_t order_2_set_min_complexity = 1;
	const size_t order_2_set_max_complexity = 5;
	const size_t order_1_set_len = 30; // MUST BE MULTIPLE OF 2

	default_random_engine re;


	auto init_sets = [&](vector<size_t>& a_train_complexity, vector<size_t>& a_test_complexity, tensor& a_train_x, tensor& a_train_y, tensor& a_test_x, tensor& a_test_y) {
		const double min_order_0_val = -10;
		const double max_order_0_val = 10;
		const double min_opd_val = -10;
		const double max_opd_val = 10;
		a_train_complexity = vector<size_t>(order_2_set_len);
		a_train_x = tensor::new_1d(order_2_set_len);
		a_train_y = tensor::new_1d(order_2_set_len);
		// INITIALIZE TRAINING DATA
		for (int i = 0; i < order_2_set_len; i++) {
			a_train_complexity[i] = random(order_2_set_min_complexity, order_2_set_max_complexity);
			auto fn = get_composite_fn(a_train_complexity[i], min_opd_val, max_opd_val, re);
			tensor x = tensor::new_2d(order_1_set_len, 1);
			tensor y = tensor::new_2d(order_1_set_len, 1);
			for (int j = 0; j < order_1_set_len; j += 2) {
				x[j][0] = random_d(min_order_0_val, max_order_0_val, re);
				y[j][0] = fn(x[j]);
				x[j + 1][0] = y[j][0];
				y[j + 1][0] = x[j][0];
			}
			a_train_x[i] = x;
			a_train_y[i] = y;
		}
		a_test_complexity = vector<size_t>(order_2_set_len);
		a_test_x = tensor::new_1d(order_2_set_len);
		a_test_y = tensor::new_1d(order_2_set_len);
		// INITIALIZE TESTING DATA
		for (int i = 0; i < order_2_set_len; i++) {
			a_test_complexity[i] = random(order_2_set_min_complexity, order_2_set_max_complexity);
			auto fn = get_composite_fn(a_test_complexity[i], min_opd_val, max_opd_val, re);
			tensor x = tensor::new_2d(order_1_set_len, 1);
			tensor y = tensor::new_2d(order_1_set_len, 1);
			for (int j = 0; j < order_1_set_len; j += 2) {
				x[j][0] = random_d(min_order_0_val, max_order_0_val, re);
				y[j][0] = fn(x[j]);
				x[j + 1][0] = y[j][0];
				y[j + 1][0] = x[j][0];
			}
			a_test_x[i] = x;
			a_test_y[i] = y;
		}
	};

	vector<size_t> train_complexity;
	vector<size_t> test_complexity;
	tensor train_x;
	tensor train_y;
	tensor test_x;
	tensor test_y;

	auto reset_sets = [&] {init_sets(train_complexity, test_complexity, train_x, train_y, test_x, test_y); };
	reset_sets();

	vector<param_mom*> pv;
	auto pmt_init = [&](ptr<param>& pmt) {
		pmt = new param_mom(0, 0, 0, 0, 0.9);
		pv.push_back((param_mom*)pmt.get());
	};

	const size_t lstm_units = 25;

	ptr<sync> s_in = new sync(order_1_set_len, pseudo::tnn({ 1, lstm_units }, pseudo::nlr(0.3), pmt_init));
	ptr<lstm> l_1 = new lstm(lstm_units, order_1_set_len, pmt_init);
	ptr<lstm> l_2 = new lstm(lstm_units, order_1_set_len, pmt_init);
	ptr<lstm> l_3 = new lstm(lstm_units, order_1_set_len, pmt_init);
	ptr<sync> s_out = new sync(order_1_set_len, pseudo::tnn({ lstm_units, 1 }, pseudo::nlr(0.3), pmt_init));
	sequential s = { s_in.get(), l_1.get(), l_2.get(), l_3.get(), s_out.get() };

	double state_structure = 0.001 / (double)pv.size();
	double learn_rate_structure = 0.02 / (double)pv.size();

	uniform_real_distribution<double> pmt_urd(-state_structure, state_structure);
	for (param_mom* pmt : pv) {
		pmt->state() = pmt_urd(re);
		pmt->learn_rate() = learn_rate_structure;
	}

	s.compile();
	s_in->unroll(order_1_set_len);
	l_1->unroll(order_1_set_len);
	l_2->unroll(order_1_set_len);
	l_3->unroll(order_1_set_len);
	s_out->unroll(order_1_set_len);

	const size_t checkpoint_interval = 1;
	const size_t mini_batch_len = 38;

	//TRAIN

	for (int epoch = 0; true; epoch++) {

		const double cost_structure = 1 / (double)mini_batch_len / (double)order_1_set_len;
		double train_cost = 0;
		for (int i = 0; i < 3; i++) {
			int ts = random(0, order_2_set_len);
			s.cycle(train_x[ts], train_y[ts]);
			train_cost += s.y_grad.abs_2d().sum_2d().sum_1d() * cost_structure;
		}

		double test_cost = 0;
		for (int ts = 0; ts < order_2_set_len; ts++) {
			s.fwd(test_x[ts]);
			s.signal(test_y[ts]);
			test_cost += s.y_grad.abs_2d().sum_2d().sum_1d() * cost_structure;
		}

		for (param_mom* pmt : pv)
			pmt->update();

		if (epoch % checkpoint_interval == 0)
			std::cout << "TRAIN: " << train_cost << " TEST: " << test_cost << std::endl;
	}

}

void genome_test() {

#pragma region MODEL
	uniform_real_distribution<double> pmt_urd(-1, 1);
	default_random_engine re(26);
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
	vector<param*> pv;
	ptr<sequential> s = pseudo::tnn({ 2, 5, 1 }, pseudo::nlr(0.3), [&](ptr<param>& pmt) {
		pmt = new param(pmt_urd(re));
		pv.push_back(pmt.get());
	});
	s->compile();
#pragma endregion
#pragma region LINK TENSOR
	tensor pmt_link_tensor = tensor::new_1d(pv.size());
	for (int i = 0; i < pmt_link_tensor.size(); i++)
		pmt_link_tensor[i].val_ptr.link(pv[i]->state_ptr);
#pragma endregion
#pragma region GET REWARD
	auto get_reward = [&](genome& a_genome) {
		pmt_link_tensor.pop(a_genome);
		double cost = 0;
		for (int i = 0; i < x.size(); i++) {
			s->fwd(x[i]);
			s->signal(y[i]);
			cost += s->y_grad.abs_1d().sum_1d();
		}
		return 1 / cost;
	};
#pragma endregion
#pragma region MUTATE
	auto mutate = [&](double val) {
		if (rand() % 100 == 0)
			return val + pmt_urd(re);
		return val;
	};
#pragma endregion
#pragma region EVOLVE
	const size_t NUM_PARENTS = 2;
	vector<genome> parents = genome(pmt_link_tensor, mutate).mutate(NUM_PARENTS);
	for (int epoch = 0; true; epoch++) {
		generation gen = generation(genome::mutate(genome::merge(parents, 40)), get_reward);
		parents = gen.best(NUM_PARENTS);
		if(epoch % 100 == 0)
			std::cout << 1 / get_reward(parents[0]) << std::endl;
	}
#pragma endregion

}

void pablo_guesser() {

	tensor x {
		{ 22, 48, 97,  2, 12, 62, 51, 90, 49, 35 },
		{ 13, 22, 49, 57, 61,  5, 17, 72, 86, 91 },
	};

	tensor y {
		{ 48, 97,  2, 12, 62, 51, 90, 49, 35, 26 },
		{ 22, 49, 57, 61,  5, 17, 72, 86, 91, 87 },
	};

	for (int i = 0; i < x.size(); i++) {
		x[i] = x[i].roll(1);
		y[i] = y[i].roll(1);
	}
	// TRAINING EXAMPLES DONE

	uniform_real_distribution<double> urd(-0.1, 0.1);
	default_random_engine re(25);

	vector<param_mom*> pv;
	auto pmt_init = [&](ptr<param>& pmt) {
		pmt = new param_mom(urd(re), 0.0002, 0, 0, 0.9);
		pv.push_back((param_mom*)pmt.get());
	};

	const int LSTM_UNITS = 35;

	ptr<sync> s_in = new sync(x.width(), pseudo::tnn({ 1, LSTM_UNITS }, pseudo::nlr(0.3), pmt_init));
	ptr<lstm> l_0 = new lstm(LSTM_UNITS, x.width(), pmt_init);
	ptr<lstm> l_1 = new lstm(LSTM_UNITS, x.width(), pmt_init);
	ptr<sync> s_out = new sync(x.width(), pseudo::tnn({ LSTM_UNITS, 1 }, pseudo::nlr(0.3), pmt_init));
	sequential seq { s_in.get(), l_0.get(), l_1.get(), s_out.get() };

	seq.compile();
	s_in->unroll(x.width());
	l_0->unroll(x.width());
	l_1->unroll(x.width());
	s_out->unroll(x.width());

	const int CHECKPOINT = 1000;

	for (int epoch = 0; true; epoch++) {

		if (epoch % CHECKPOINT == 0) {
			system("cls");
			printf("\033[%d;%dH", 0, 0);
		}

		for (int ts = 0; ts < x.height(); ts++) {
			seq.cycle(x[ts], y[ts]);
			if (epoch % CHECKPOINT == 0) {
				std::cout << y[ts].to_string() << std::endl;
				std::cout << seq.y.to_string() << std::endl << std::endl;
			}
		}


		for (param_mom* pmt : pv)
			pmt->update();

	}

}

class mm {
public:
	mm() {

	}
	mm(double a_state, double a_gamma, double a_beta) {
		m_s = a_state;
		m_gamma = a_gamma;
		m_beta = a_beta;
	}

	double sign(double x) {
		if (x >= 0) return 1;
		else return -1;
	}
	double& state() { return m_s; }
	double& gamma() { return m_gamma; }
	double alpha() { return 1 - m_beta; }
	double& beta() { return m_beta; }
	double& momentum() { return m_momentum; }
	void update(double a_c) {
		m_ds = gamma() * (beta() * momentum() + alpha() * a_c);
		m_s += m_ds;
	}
	void reward(double a_R) {
		m_dR = a_R - m_R;
		m_R = a_R;
		double slope = (m_dR * m_ds);
		m_momentum = beta() * momentum() + alpha() * sign(slope);
	}
	void change_reward(double a_dR) {
		m_dR = a_dR;
		m_R = m_R + m_dR;
		double slope = (m_dR * m_ds);
		m_momentum = beta() * momentum() + alpha() * sign(slope);
	}

private:
	double m_s = 0;
	double m_ds = 0;
	double m_R = 0;
	double m_dR = 0;
	double m_gamma = 0;
	double m_beta = 0;
	double m_momentum = 0;

};

void test_mm() {

	tensor x = {
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
		{1, 2},
		{3, 4}
	};

	tensor y = {
		{0},
		{1},
		{1},
		{0},
		{5},
		{12},
	};

	uniform_real_distribution<double> urd(-1, 1);
	default_random_engine re(35);

	vector<param*> pv;

	auto pmt_init = [&](ptr<param>& pmt) {
		pmt = new param(urd(re));
		pv.push_back(pmt.get());
	};

	ptr<model> s = pseudo::tnn({ 2, 5, 1 }, pseudo::nlr(0.3), pmt_init);
	s->compile();

	const double GAMMA = 0.002;
	const double BETA = 0.9;

	vector<mm> pmm = vector<mm>(pv.size());
	for (int i = 0; i < pv.size(); i++)
		pmm[i] = mm(pv[i]->state(), GAMMA, BETA);

	std::normal_distribution<double> nd(0, 1);

	auto get_rcv = [&]() {
		/*if (rand() % pv.size() == 0)
			return nd(re);
		else*/
			return nd(re);
	};

	auto get_reward = [&]() {
		double cost = 0;
		for (int ts = 0; ts < x.size(); ts++) {
			s->fwd(x[ts]);
			s->signal(y[ts]);
			cost += s->y_grad.abs_1d().sum_1d();
		}
		return 1 / cost;
	};

	for (int epoch = 0; true; epoch++) {
		for (int i = 0; i < pmm.size(); i++) {
			pmm[i].update(get_rcv());
			pv[i]->state() = pmm[i].state();
		}
		double reward = get_reward();
		double cost = 1 / reward;
		for (int i = 0; i < pmm.size(); i++)
			pmm[i].reward(reward);

		if(epoch % 10000 == 0)
			for (int i = 0; i < pmm.size(); i++) {
				pmm[i].gamma() = 0.002 * std::tanh(cost);
			}

		if (epoch % 10000 == 0) {
			std::cout << "REWARD: " << reward << ", COST: " << 1 / reward << std::endl;
		}
	}

}

void test_srtt() {

	default_random_engine re(27);
	std::normal_distribution<double> pmt_nd(0, 0.1);

	vector<param*> pv_0;
	auto pmt_0_init = [&](ptr<param>& pmt) {
		pmt = new param(pmt_nd(re));
		pv_0.push_back(pmt.get());
	};

	vector<param*> pv_1;
	auto pmt_1_init = [&](ptr<param>& pmt) {
		pmt = new param(pmt_nd(re));
		pv_1.push_back(pmt.get());
	};

	ptr<model> m_0 = pseudo::tnn({ 1, 10, 1 }, pseudo::nlr(0.3), pmt_0_init);
	ptr<model> m_1 = pseudo::tnn({ 2, 64, pv_0.size() }, pseudo::nth(), pmt_1_init);

	m_0->compile();
	m_1->compile();

	tensor order_0_states = tensor::new_1d(pv_0.size());
	for (int i = 0; i < order_0_states.size(); i++)
		order_0_states[i].val_ptr.link(pv_0[i]->state_ptr);
	tensor order_0_save = order_0_states.clone();

	const double GAMMA = 0.00002;
	const double BETA = 0.99;

	vector<mm> mm_1 = vector<mm>(pv_1.size());
	for (int i = 0; i < mm_1.size(); i++)
		mm_1[i] = mm(pv_1[i]->state(), GAMMA, BETA);

	const size_t ORDER_0_MAX_COMPLEXITY = 1;
	const size_t ORDER_0_MAX_TRAINING_SETS = 300;
	const size_t ORDER_0_MAX_TESTING_SETS = 100;
	uniform_real_distribution<double> ts_0_urd(-10, 10);

	std::normal_distribution<double> training_nd(0, 1);

	auto get_rcv = [&]() {
		return training_nd(re);
	};

	for (int epoch = 0; true; epoch++) {
		
		for (int i = 0; i < mm_1.size(); i++) {
			mm_1[i].update(get_rcv());
			pv_1[i]->state() = mm_1[i].state();
		}

		composite_function func = get_composite_fn(ORDER_0_MAX_COMPLEXITY, -10, 10, re);
		
		order_0_states.pop(order_0_save); // RELOAD ORDER 0 STATES FROM START

		for (int ts = 0; ts < ORDER_0_MAX_TRAINING_SETS; ts++) {
			double x = ts_0_urd(re);
			double y = func(x);
			m_1->x[0].val() = x;
			m_1->x[1].val() = y;
			m_1->fwd();
			order_0_states.add_1d(m_1->y, order_0_states);
		}

		double cost = 0;

		for (int ts = 0; ts < ORDER_0_MAX_TESTING_SETS; ts++) {
			double x = ts_0_urd(re);
			double y = func(x);
			m_0->x[0].val() = x;
			m_0->fwd();
			cost += abs(m_0->y[0] - y);
		}

		double reward = 1 / cost;

		for (int i = 0; i < mm_1.size(); i++)
			mm_1[i].reward(reward);

		if (epoch % 1000 == 0) {
			std::cout << cost << std::endl;
			for (int i = 0; i < mm_1.size(); i++) {
				mm_1[i].gamma() *= 0.7;
			}
		}

	}
		
}

void self_aware() {

	uniform_real_distribution<double> pmt_urd(-1, 1);

	vector<param*> pv;

	auto pmt_init = [&](ptr<param>& pmt) {
		pmt = new param(pmt_urd(aurora::static_vals::aurora_random_engine));
		pv.push_back(pmt.get());
	};

	ptr<model> m = pseudo::tnn({ 2, 5, 2 }, pseudo::nlr(0.3), pmt_init);
	m->compile();

	tensor pmt_link = tensor::new_1d(pv.size());
	for (int i = 0; i < pmt_link.size(); i++)
		pmt_link[i].val_ptr.link(pv[i]->state_ptr);

	tensor order_0_x = m->x[0].link();
	tensor order_0_y_hat = m->y[0].link();
	tensor order_1_x = m->x[1].link();
	tensor order_1_param_index = m->y[0].link();
	tensor order_1_param_state = m->y[1].link();

	uniform_real_distribution<double> pmt_x_0(-10, 10);

	auto get_reward = [&]() {
		order_0_x.val() = 0;
		order_1_x.val() = 1;
		m->fwd();

		pmt_link[(int)order_1_param_index].val() = order_1_param_state.val();
		double cost = 0;
		for (int ts = 0; ts < 10; ts++) {
			double x = pmt_x_0(aurora::static_vals::aurora_random_engine);
			double y = x + 3;
			order_0_x.val() = x;
			order_1_x.val() = 0;
			m->fwd();
			cost += abs(order_0_y_hat - y);
		}
		double reward = 1 / cost;

		return reward;
	};

	for (int epoch = 0; epoch < 10000; epoch++) {

	}

}

void proc_generator() {

	tensor train_x {
		{{0,0,4,0,0,0,1.722584,1,9.860078,},
		{-1.8,-0.77,-0.53,-0.7071068,0,0,1,1.5289,1,},
		{-2.11,0.68,-0.754,-0.7071068,0,0,1,1.5289,1,},
		{2.11,0.68,-0.754,-0.7071068,0,0,1,1.5289,1,},
		{1.8,-0.77,-0.53,-0.7071068,0,0,1,1.5289,1,},
		{0,0,0,0,0,0,4.043402,1.846521,2.5525,},
		},
		{{0,0,4,0,0,0,1.722584,1,5.540674,},
		{-2.26,-0.18,-1.3784,-0.7071068,0,0,1,2.482169,1,},
		{-2.27,0.55,-0.60562,-0.7071068,0,0,1,2.482169,1,},
		{2.27,0.55,-0.60562,-0.7071068,0,0,1,2.482169,1,},
		{2.26,-0.18,-1.3784,-0.7071068,0,0,1,2.482169,1,},
		{0,0,0,0,0,0,4.043402,1.846521,2.5525,},
		},
		{{0,0,4,0,0,0,1.722584,1.846521,5.540674,},
		{-2.26,-0.18,0.5400001,-0.7071068,0,0,1,2.482169,1,},
		{-2.27,0.55,0,-0.7071068,0,0,1,2.482169,1,},
		{2.27,0.55,0,-0.7071068,0,0,1,2.482169,1,},
		{2.26,-0.18,0.54,-0.7071068,0,0,1,2.482169,1,},
		{0,0,0,0,0,0,4.043402,1.846521,2.5525,},
		},
		{{0,0,2.63,0,0,0,1.722584,1.846521,5.920644,},
		{-2.26,-1.01,-1,-0.7071068,0,0,1,2,1,},
		{-2.27,1.27,-1,-0.7071068,0,0,1,2,1,},
		{2.27,1.27,-1,-0.7071068,0,0,1,2,1,},
		{2.26,-1.01,-1,-0.7071068,0,0,1,2,1,},
		{0,0,0,0,0,0,4.043402,1.846521,2.5525,},
		},
		{{0,0,1.11,0,0,0,1.722584,1.846521,1.559083,},
		{-2,-1,-1,-0.7071068,0,0,0.5,2,0.5,},
		{-2,1,-1,-0.7071068,0,0,0.5,2,0.5,},
		{2,1,-1,-0.7071068,0,0,0.5,2,0.5,},
		{2,-1,-1,-0.7071068,0,0,0.5,2,0.5,},
		{0,0,0,0,0,0,4.043402,2.199021,1.148651,},
		},
	};
	tensor test_x{
		{{0,0,3.7,0,0,0,1.722584,2.661576,7.680356,},
		{-2,-1,-1,-0.7071068,0,0,1.5,2,1.5,},
		{-2,1,-1,-0.7071068,0,0,1.5,2,1.5,},
		{2,1,-1,-0.7071068,0,0,1.5,2,1.5,},
		{2,-1,-1,-0.7071068,0,0,1.5,2,1.5,},
		{0,0,0,0,0,0,4.043402,2.199021,1.148651,},
		},
		{{0,0,2.82,0,0,0,1.722584,2.661576,5.722557,},
		{-2,-1,0.53,-0.7071068,0,0,1.5,2,1.5,},
		{-2,1,0.53,-0.7071068,0,0,1.5,2,1.5,},
		{2,1,0.53,-0.7071068,0,0,1.5,2,1.5,},
		{2,-1,0.53,-0.7071068,0,0,1.5,2,1.5,},
		{0,0,0,0,0,0,4.043402,2.199021,1.148651,},
		},
		{{0,0,2.82,0,0,0,1.722584,0.6529909,5.722557,},
		{-2,-0.45,0.53,-0.7071068,0,0,0.3,2,0.3,},
		{-2,0.45,0.53,-0.7071068,0,0,0.3,2,0.3,},
		{2,0.45,0.53,-0.7071068,0,0,0.3,2,0.3,},
		{2,-0.45,0.53,-0.7071068,0,0,0.3,2,0.3,},
		{0,0,0,0,0,0,4.043402,0.8935723,1.148651,},
		},
		{{0,0,2.56,0,0,0,1.722584,0.6529909,4.349715,},
		{-2,-0.45,0.53,-0.7071068,0,0,0.3,1.24754,0.3,},
		{-2,0.45,0.53,-0.7071068,0,0,0.3,1.24754,0.3,},
		{2,0.45,0.53,-0.7071068,0,0,0.3,1.24754,0.3,},
		{2,-0.45,0.53,-0.7071068,0,0,0.3,1.24754,0.3,},
		{0,0,0,0,0,0,4.043402,0.8935723,1.148651,},
		},
		{{0,0,2.56,0,0,0,1.722584,0.6529909,4.349715,},
		{-0.9,-0.45,0.53,-0.7071068,0,0,0.3,1.24754,0.3,},
		{-0.9,0.45,0.53,-0.7071068,0,0,0.3,1.24754,0.3,},
		{0.9,0.45,0.53,-0.7071068,0,0,0.3,1.24754,0.3,},
		{0.9,-0.45,0.53,-0.7071068,0,0,0.3,1.24754,0.3,},
		{0,0,0,0,0,0,1.972978,0.8935723,1.148651,},
		},
	};

	for (int i = 0; i < train_x.size(); i++)
		train_x[i] = train_x[i].unroll();
	for (int i = 0; i < test_x.size(); i++)
		test_x[i] = test_x[i].unroll();

	uniform_real_distribution<double> pmt_urd(-0.1, 0.1);
	default_random_engine re(27);

	vector<param_mom*> pv;

	auto pmt_init = [&](ptr<param>& pmt) {
		pmt = new param_mom(pmt_urd(re), 0.002, 0, 0, 0.9);
		pv.push_back((param_mom*)pmt.get());
	};

	ptr<sequential> compressor = pseudo::tnn({ 54, 30, 5, 30, 54 }, pseudo::nlr(0.3), pmt_init);
	ptr<model> second_half = new sequential(vector<ptr<model>>(compressor->models.begin() + 4, compressor->models.end()));
	compressor->compile();
	second_half->compile();

	const int CHECKPOINT = 1000;

	for (int epoch = 0; epoch / CHECKPOINT < 100; epoch++) {

		double train_cost = 0;
		double test_cost = 0;

		for (int ts = 0; ts < train_x.size(); ts++) {
			compressor->cycle(train_x[ts], train_x[ts]);
			if(epoch % CHECKPOINT == 0)
				train_cost += compressor->y_grad.abs_1d().sum_1d();
		}
		if(epoch % CHECKPOINT == 0)
			for (int ts = 0; ts < test_x.size(); ts++) {
				compressor->cycle(test_x[ts], test_x[ts]);
				if (epoch % CHECKPOINT == 0)
					test_cost += compressor->y_grad.abs_1d().sum_1d();
				std::cout << "COMPRESSED: " << second_half->x.to_string() << std::endl;
			}

		if (epoch % CHECKPOINT == 0)
			std::cout << "TRAIN: " << train_cost << ", TEST: " << test_cost << std::endl;

		for (param_mom* pmt : pv)
			pmt->update();
	}



}

void cnl_test() {

	const size_t X_HEIGHT = 10;
	const size_t X_WIDTH = 10;

	tensor x = {
		tensor::new_2d(X_HEIGHT, X_WIDTH, 0),
		tensor::new_2d(X_HEIGHT, X_WIDTH, 1),
		tensor::new_2d(X_HEIGHT, X_WIDTH, 2),
	};

	uniform_real_distribution<double> pmt_urd(-1, 1);

	vector<param_mom*> pv;
	auto pmt_init = [&](ptr<param>& pmt) {
		pmt = new param_mom(pmt_urd(aurora::static_vals::aurora_random_engine), 0.0002, 0, 0, 0.9);
		pv.push_back((param_mom*)pmt.get());
	};
	
	ptr<cnl> c1 = new cnl(X_HEIGHT, X_WIDTH, 2, 2, 1, pmt_init);
	ptr<layer> l1 = new layer(c1->y_strides(), new layer(c1->x_strides(), pseudo::nth(), pmt_init));
	ptr<cnl> c2 = new cnl(c1->y_strides(), c1->x_strides(), 2, 2, 1, pmt_init);
	ptr<layer> l2 = new layer(c2->y_strides(), new layer(c2->x_strides(), pseudo::nth(), pmt_init));
	ptr<cnl> c3 = new cnl(c2->y_strides(), c2->x_strides(), 2, 2, 1, pmt_init);

	ptr<sequential> s = new sequential {
		c1.get(),
		l1.get(),
		c2.get(),
		l2.get(),
		c3.get(),
	};

	std::cout << "COMPILING MODEL" << std::endl;
	s->compile();

	tensor y = {
		tensor::new_2d(c3->y_strides(), c3->x_strides(), 3),
		tensor::new_2d(c3->y_strides(), c3->x_strides(), 2),
		tensor::new_2d(c3 ->y_strides(), c3->x_strides(), 1),
	};

	c1->unroll(X_HEIGHT, X_WIDTH);
	c2->unroll(c1->y_strides(), c1->x_strides());
	c3->unroll(c2->y_strides(), c2->x_strides());

	std::cout << "TRAINING MODEL" << std::endl;
	for (int epoch = 0; true; epoch++) {

		double cost = 0;
		for (int ts = 0; ts < x.size(); ts++) {
			s->cycle(x[ts], y[ts]);
			cost += s->y_grad.abs_2d().sum_2d().sum_1d();
			if(epoch % 100 == 0)
				std::cout << s->y.to_string() << std::endl;
		}

		if (epoch % 100 == 0)
			std::cout << cost << std::endl;

		for (param_mom* pmt : pv)
			pmt->update();

	}

}

void auto_encoder() {

	const size_t H_LEN = 100;
	const size_t X_LEN = H_LEN * 8 + 1;
	const size_t NUM_TRAINING_SETS = 1000;
	tensor x = tensor::new_2d(NUM_TRAINING_SETS, X_LEN);

	for (int i = 0; i < x.size(); i++)
		for (int j = 0; j < x[i].size(); j++)
			x[i][j] = (rand() % 256 - 128) / (double)64;

	uniform_real_distribution<double> pmt_urd(-1, 1);
	default_random_engine re(27);

	vector<param_mom*> pv;
	auto pmt_init = [&](ptr<param>& pmt) {
		pmt = new param_mom(pmt_urd(re), 0.0002 / NUM_TRAINING_SETS, 0, 0, 0.9);
		pv.push_back((param_mom*)pmt.get());
	};

	ptr<model> m = pseudo::tnn({ X_LEN, X_LEN / 2, H_LEN, X_LEN / 2, X_LEN }, pseudo::nlr(0.3), pmt_init);
	m->compile();

	const string file_name = "auto_encoder";

	const bool IMPORT_FROM_FILE = true;

	if (IMPORT_FROM_FILE)
		pl_import_from_file(file_name, pv);

	const int CHECKPOINT_INTERVAL = 100;

	for (int epoch = 0; true; epoch++) {
		double cost = 0;
		for (int ts = 0; ts < x.size(); ts++) {
			m->cycle(x[ts], x[ts]);
			cost += m->y_grad.abs_1d().sum_1d();
		}
		for (param_mom* pmt : pv)
			pmt->update();
		if (epoch % CHECKPOINT_INTERVAL == 0) {
			std::cout << "COST: " << cost / NUM_TRAINING_SETS * 255 << std::endl;
			std::cout << "DES: " << x[x.size() - 1].to_string() << std::endl;
			std::cout << "ACT: " << m->y.to_string() << std::endl;
			pl_export_to_file(file_name, pv);
		}
	}

}

int main() {

	srand(time(NULL));

	auto_encoder();

	return 0;

}