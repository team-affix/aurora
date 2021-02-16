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

	uniform_real_distribution<double> urd(-1, 1);
	default_random_engine re(25);

	ptr<sequential> s = pseudo::tnn({ 2, 17, 1 }, pseudo::nlr(0.3), [&](ptr<param>& pmt) {
		pmt = new param_mom(urd(re), 0.02, 0, 0, 0.9);
		pl.push_back((param_mom*)pmt.get());
	});

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

	sync s = sync(pseudo::tnn({ 2, 25, 1 }, pseudo::nlr(0.3), [&](ptr<param>& pmt) {
		pmt = new param_sgd_mt(urd(re), 0.0002, 0);
		pl.push_back((param_sgd_mt*)pmt.get());
	}));

	s.prep(numClones);
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
	ptr<sync> s_prev = new sync(pseudo::tnn({ 2, 5, 1 }, pseudo::nlr(0.3), [&](ptr<param>& pmt) {
		pmt = new param_sgd(urd(re), 0.02, 0);
		pl.push_back((param_sgd*)pmt.get());
	}));
	ptr<sync> s = (sync*)s_prev->clone();

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

	sync* s0 = new sync(pseudo::tnn({ 2, lstm_units }, pseudo::nlr(0.3), pmt_init));
	lstm* l = new lstm(lstm_units, pmt_init);
	sync* s1 = new sync(pseudo::tnn({ lstm_units, 1 }, pseudo::nlr(0.3), pmt_init));

	ptr<sync> s0_ptr = s0;
	ptr<lstm> l0_ptr = l;
	ptr<sync> s1_ptr = s1;

	sequential s = sequential({ s0, l, s1 });

	s0->prep(4);
	l->prep(4);
	s1->prep(4);

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
	ptr<sync> s_0 = new sync(pseudo::tnn({ 2, 5, 1 }, pseudo::nlr(0.3), [&](ptr<param>& pmt) {
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
	s_0->prep(x_0.size());
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
	ptr<sync> s_0 = new sync(pseudo::tnn({ 2, 5, 1 }, pseudo::nlr(0.3), [&](ptr<param>& pmt) {
		pmt = new param_mom(urd(re), learn_rate, 0, 0, beta);
		pv_0.push_back((param_mom*)pmt.get());
		}));
#pragma endregion
#pragma region ORDER 1 INSTANTIATE
	const int x_1_height = 10;
	tensor x_1 = tensor::new_2d(x_1_height, 1, urd, re);
	vector<param_mom*> pv_1;
	ptr<sequential> s_1 = pseudo::tnn({ 1, pv_0.size() / 2, pv_0.size() }, pseudo::nlr(0.3), [&](ptr<param>& pmt) {
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
	s_0->prep(x_0.size());
	s_0->compile();
	s_0->unroll(x_0.size());
	s_1->compile();
#pragma endregion
#pragma region CONFIGURE TRAINING
	const double min_ts_cost = 11;
	const int checkpoint_interval = 100;
	const int deviant_update_amount = 10;
#pragma endregion
#pragma region GET ORDER 1 TRAINING SET Y
	tensor y_1 = tensor::new_2d(x_1.size(), pv_0.size(), urd, re);
	auto get_tsx_1 = [&](tensor tsy_1) {
		pv_0_states.pop(tsy_1);
		s_0->fwd(x_0);
		s_0->signal(y_0);
		return s_0->y_grad.abs_2d().sum_2d();
	};
	for (int tsIndex = 0; tsIndex < x_1.size(); tsIndex++) {
		x_1[tsIndex] = get_tsx_1(y_1[tsIndex]);
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
			s_1->fwd(deviant_x);

			// POP BACK OF ORDER 1 TS VECTORS
			x_1.vec().pop_back();
			y_1.vec().pop_back();

			// APPEND NEW TRAINING SET
			x_1.vec().push_back(get_tsx_1(s_1->y));
			y_1.vec().push_back(s_1->y);

		}

		if (epoch % checkpoint_interval == 0 && cost_low)
			std::cout << "TEST COST: " << get_tsx_1(s_1->fwd(deviant_x)).abs_1d().sum_1d().val() << std::endl;

		update_params(pv_1);
	}
}

void zil_trader() {

	uniform_real_distribution<double> urd(-1, 1);
	uniform_real_distribution<double> urd2(0.1, 3);
	uniform_real_distribution<double> urd3(-2, 2);
	default_random_engine re(8040);

	tensor value_tensor_2d = tensor::new_2d(2, 20, urd2, re);
	tensor value_tensor_3d = tensor::new_1d(value_tensor_2d.size());
	for (int i = 0; i < value_tensor_3d.size(); i++)
		value_tensor_3d[i] = value_tensor_2d[i].roll(1);

	vector<param_mom*> pv;
	auto pmt_init = [&](ptr<param>& pmt) {
		pmt = new param_mom(urd(re), 0.2, 0, 0, 0.9);
		pv.push_back((param_mom*)pmt.get());
	};

	size_t lstm_units = 50;
	size_t episode_len = value_tensor_2d.width();
	size_t training_sets = value_tensor_2d.height();

	ptr<sync> s0 = new sync(pseudo::tnn({ 1, lstm_units }, pseudo::nlr(0.3), pmt_init));
	ptr<lstm> l0 = new lstm(lstm_units, pmt_init);
	ptr<lstm> l1 = new lstm(lstm_units, pmt_init);
	ptr<lstm> l2 = new lstm(lstm_units, pmt_init);
	ptr<sync> s1 = new sync(pseudo::tnn({ lstm_units, 2 }, { pseudo::nlr(0.3), pseudo::nth() }, pmt_init));
	ptr<sequential> seq = new sequential({ s0.get(), l0.get(), l1.get(), l2.get(), s1.get() });
	sync s = sync(seq.get());

	s0->prep(episode_len);
	l0->prep(episode_len);
	l1->prep(episode_len);
	l2->prep(episode_len);
	s1->prep(episode_len);
	s0->compile();
	l0->compile();
	l1->compile();
	l2->compile();
	s1->compile();
	s0->unroll(episode_len);
	l0->unroll(episode_len);
	l1->unroll(episode_len);
	l2->unroll(episode_len);
	s1->unroll(episode_len);
	s.prep(training_sets);
	s.compile();
	s.unroll(training_sets);


	auto get_actions = [](tensor& action_tensor) {
		vector<bool> actions = vector<bool>(action_tensor.size());
		for (int i = 0; i < action_tensor.size(); i++) {
			tensor action = action_tensor[i]; // WILL BE IN FORM {(ACT), (DO NOTHING)}
			double act_certainty = action[0].val();
			double no_certainty = action[1].val();
			actions[i] = act_certainty > no_certainty;
		}
		return actions;
	};

	auto get_earnings = [&](tensor& value_tensor_1d, vector<bool>& actions) {
		double zil = 0;
		double usd = 1;
		// BUY FIRST, THEN SELL
		bool purchasing = true;
		double purchase_price = 0;
		for (int i = 0; i < actions.size(); i++)
			if (actions[i]) {
				double current_price = value_tensor_1d[i].val();
				if (purchasing) {
					purchase_price = current_price;
					zil = usd / current_price;
				}
				else {
					usd = zil * current_price;
				}
				purchasing = !purchasing;
			}
		return usd - 1;
	};

	auto des_success = [](vector<bool>& actions) {
		tensor result = tensor::new_2d(actions.size(), 2);
		for (int i = 0; i < actions.size(); i++)
			if (actions[i])
				result[i][0] = 1;
			else
				result[i][1] = 1;
		return result;
	};

	auto des_failure = [](vector<bool>& actions) {
		tensor result = tensor::new_2d(actions.size(), 2);
		for (int i = 0; i < actions.size(); i++)
			if (actions[i])
				result[i][0] = -1;
			else
				result[i][1] = -1;
		return result;
	};

	auto des = [](double earnings, vector<bool>& actions) {
		tensor result = tensor::new_2d(actions.size(), 2);
		for (int i = 0; i < actions.size(); i++)
			if (actions[i])
				result[i][0] = std::tanh(earnings * 0.01);
			else
				result[i][1] = std::tanh(earnings* 0.01);
		return result;
	};

	double prev_epoch_earnings = 0;

	vector<vector<bool>> epoch_actions = vector<vector<bool>>(training_sets);

	tensor action_current = tensor::new_2d(episode_len, 2, urd, re);
	tensor action_momentum = tensor::new_2d(episode_len, 2, 0);
	tensor action_beta = tensor::new_2d(episode_len, 2, 0.9);
	tensor action_beta_compliment = tensor::new_2d(episode_len, 2, 0.1);

	auto update_action_momentum = [&](tensor& rct) {
		action_momentum = action_momentum.mul_2d(action_beta).add_2d(action_beta_compliment.mul_2d(rct));

	};

	for (int epoch = 0; true; epoch++) {
		while (true) {
			tensor test_tensor = action_current.clone();
			for (int i = 0; i < epoch && i < 5; i++) {
				int x = rand() % episode_len;
				int y = rand() % 2;
				test_tensor[x][y].val() += urd(re);
			}
			test_tensor.tanh_2d(test_tensor);
			vector<bool> actions = get_actions(test_tensor);
			double epoch_earnings = get_earnings(value_tensor_2d[1], actions);
			/*if (epoch_earnings > 6699)
			{
				std::cout << value_tensor_2d[0].to_string() << std::endl;
				for (int i = 0; i < actions.size(); i++)
					std::cout << actions[i] << std::endl;
				return;
			}*/
			if (epoch_earnings > prev_epoch_earnings) {
				prev_epoch_earnings = epoch_earnings;
				action_current = test_tensor;
				break;
			}
		}
		std::cout << prev_epoch_earnings << std::endl;
	}



	for (int epoch = 0; true; epoch++) {

		double epoch_earnings = 0;

		for (int tsIndex = 0; tsIndex < value_tensor_3d.size(); tsIndex++) {
			ptr<model> m = s.unrolled[tsIndex];
			tensor& ts = value_tensor_3d[tsIndex];
			m->fwd(ts);
			tensor rcv = tensor::new_2d(episode_len, 2, urd, re);
			tensor div = tensor::new_2d(episode_len, 2, (0.01 * epoch) + 0.1);
			tensor action_tensor = m->y.add_2d(rcv.div_2d(div)).tanh_2d();
			vector<bool> actions = get_actions(action_tensor);
			epoch_earnings += get_earnings(value_tensor_2d[tsIndex], actions);
			epoch_actions[tsIndex] = actions;
		}

		if (epoch_earnings >= prev_epoch_earnings) {
			prev_epoch_earnings = epoch_earnings;
			for (int tsIndex = 0; tsIndex < value_tensor_3d.size(); tsIndex++) {
				s.unrolled[tsIndex]->signal(des_success(epoch_actions[tsIndex]));
				s.unrolled[tsIndex]->bwd();
			}
		}
		else {
			for (int tsIndex = 0; tsIndex < value_tensor_3d.size(); tsIndex++) {
				s.unrolled[tsIndex]->signal(des_failure(epoch_actions[tsIndex]));
				s.unrolled[tsIndex]->bwd();
			}
		}

		/*for (int tsIndex = 0; tsIndex < value_tensor_3d.size(); tsIndex++) {
			s.unrolled[tsIndex]->signal(des(epoch_earnings, epoch_actions[tsIndex]));
			s.unrolled[tsIndex]->bwd();
		}*/

		if (epoch % 10 == 0)
			std::cout << epoch_earnings << std::endl;

		for (param_mom* pmt : pv) {
			pmt->momentum() = pmt->beta() * pmt->momentum() + (1 - pmt->beta()) * pmt->gradient();
			pmt->state() -= pmt->learn_rate() * pmt->momentum();
			pmt->gradient() = 0;
		}

	}

}



int main() {

	srand(time(NULL));

	zil_trader();

	return 0;

}