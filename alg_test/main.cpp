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
using namespace affix_base::data;
using namespace aurora::maths;
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

	{
		tensor mat_3 = tensor::new_2d(10, 10);
		tensor mat_4;
		{
			tensor mat_5 = mat_3.range_2d(0, 0, 2, 2);
			mat_4.group_join(mat_5);
		}
		mat_4.pop(tensor::new_2d(2, 2, 1));
		assert(mat_3[0][0] == 1);
		assert(mat_3[0][1] == 1);
		assert(mat_3[1][0] == 1);
		assert(mat_3[1][1] == 1);
	}

	{
		tensor t1 = { 1, 2, 3, 4 };
		tensor t2 = t1.range(0, 2);
		tensor t3 = { 5, 6 };
		t2[0].group_join_all_ranks(t3[0]);
		assert(t1[0] == 5);
	}

	{
		tensor t1 = { 0, 0, 0, 0 };
		tensor t2 = t1.range(0, 2);

	}

	tensor vec_3;
	tensor vec_4 = tensor::new_1d(2);
	{
		tensor t1 = tensor::new_1d(1);
		tensor t2 = tensor::new_1d(1);
		vec_3 = t1.cat(t2);
	}
	vec_3.group_join_all_ranks(vec_4);

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

	Sequential s = pseudo::tnn({ 2, 5, 1 }, pseudo::nlr(0.3));
	s->param_recur([&](Param& pmt) {
		pmt = new param_mom(urd(re), 0.02, 0, 0, 0.9);
		pl.push_back((param_mom*)pmt.get());
	});
	s->compile();

	printf("");

	const size_t checkpoint_interval = 10000;

	for (int epoch = 0; epoch < 1000000; epoch++) {

		if (epoch % checkpoint_interval == 0) {
			printf("\033[%d;%dH", 0, 0);
			std::cout << epoch << std::endl;
		}

		for (int tsIndex = 0; tsIndex < x.size(); tsIndex++) {
			s->cycle(x[tsIndex], y[tsIndex]);
			if (epoch % checkpoint_interval == 0)
				std::cout << x[tsIndex].to_string() << " " << s->y.to_string() << std::endl;
		}

		for (param_mom* pmt : pl)
			pmt->update();
	}

	for (param_mom* pmt : pl) {
		std::cout << pmt->state() << std::endl;
	}
}

void tnn_compiled_xor_test() {
	
	param_vector pv;
	Model s = basic::tnn({ 2, 5, 1 }, pv);
	
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

	const size_t checkpoint_interval = 10000;

	for (int epoch = 0; epoch < 1000000; epoch++) {

		if (epoch % checkpoint_interval == 0) {
			printf("\033[%d;%dH", 0, 0);
			std::cout << epoch << std::endl;
		}

		for (int tsIndex = 0; tsIndex < x.size(); tsIndex++) {
			s->cycle(x[tsIndex], y[tsIndex]);
			if (epoch % checkpoint_interval == 0)
				std::cout << x[tsIndex].to_string() << " " << s->y.to_string() << std::endl;
		}

		pv.update();

	}

	for (Param& pmt : pv) {
		std::cout << pmt->state() << std::endl;
	}

}

void tanh_test() {

	ptr<models::tanh> p = new models::tanh(1, 1, 0);
	p->compile();
	tensor x = 0;
	assert(p->fwd(x) == 0);
	x = 1;
	assert(p->fwd(x) == std::tanh(1));
	p = new models::tanh(1, 1, 1);
	x = -999999;
	assert(p->fwd(x) == 0);
	x = 9999999;
	assert(p->fwd(x) == 2);

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

	Sequential s = pseudo::tnn({ 2, 17, 1 }, pseudo::nlr(0.3));
	s->param_recur(PARAM_INIT(param_mom(urd(aurora::static_vals::random_engine), 0.0002, 0, 0, 0.9), pl));
	s->compile();
	
	for (int epoch = 0; epoch < 1000000; epoch++) {

		if (epoch % 10000 == 0) {
			printf("\033[%d;%dH", 0, 0);
			std::cout << epoch << std::endl;
		}

		double cost = 0;

		for (int tsIndex = 0; tsIndex < x.size(); tsIndex++) {
			s->cycle(x[tsIndex], y[tsIndex]);
			cost += s->y_grad.abs_1d().sum_1d();
		}

		if (epoch % 10000 == 0) {
			std::cout << cost << std::endl;
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

	sync s = sync(pseudo::tnn({ 2, 25, 1 }, pseudo::nlr(0.3)));
	s.param_recur([&](Param& pmt) {
		pmt = new param_sgd_mt(urd(re), 0.0002, 0);
		pl.push_back((param_sgd_mt*)pmt.get());
	});
	s.prep(numClones);
	s.compile();
	s.unroll(numClones);
	
	persistent_thread threads[numClones];

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
			threads[tsIndex].call([&] {
				seq.cycle(seq_x, seq_y);
			});
		}

		for (persistent_thread& thd : threads) {
			thd.join_call();
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
	Sync s_prev = new sync(pseudo::tnn({ 2, 5, 1 }, pseudo::nlr(0.3)));
	s_prev->prep(4);

	Sync s = (sync*)s_prev->clone();
	
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
	Sequential s = pseudo::tnn({ x_order_1_len, h_1_len, h_2_len, h_3_len, encoded_len, h_3_len, h_2_len, h_1_len, x_order_1_len }, pseudo::nlr(0.3));
	s->param_recur([&](Param& pmt) {
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
				epoch_cost += s->y_grad.abs_1d().sum_1d();
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

	auto pmt_init = [&](Param& pmt) {
		pmt = new param_sgd(urd(re), 0.02, 0);
		pv.push_back(pmt);
	};

	sync* s0 = new sync(pseudo::tnn({ 2, lstm_units }, pseudo::nlr(0.3)));
	lstm* l = new lstm(lstm_units);
	sync* s1 = new sync(pseudo::tnn({ lstm_units, 1 }, pseudo::nlr(0.3)));

	Sync s0_ptr = s0;
	ptr<lstm> l0_ptr = l;
	Sync s1_ptr = s1;

	sequential s = sequential({ s0, l, s1 });
	s.param_recur(pmt_init);

	s0->prep(4);
	l->prep(4);
	s1->prep(4);

	s.compile();

	s0->unroll(4);
	l->unroll(4);
	s1->unroll(4);

	const int checkpoint_interval = 10000;

	for (int epoch = 0; epoch < 100000; epoch++) {
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
	Sync s_0 = new sync(pseudo::tnn({ 2, 5, 1 }, pseudo::nlr(0.3)));
	s_0->param_recur([&](Param& pmt) {
		pmt = new param_mom(urd(re), learn_rate, 0, 0, beta);
		pv_0.push_back((param_mom*)pmt.get());
	});
#pragma endregion
#pragma region ORDER 1 INSTANTIATE
	const int x_1_height = 20;
	tensor x_1 = tensor::new_2d(x_1_height, pv_0.size(), urd, re);
	vector<param_mom*> pv_1;
	Sequential s_1 = pseudo::tnn({ pv_0.size(), pv_0.size() * 2, 1 }, pseudo::nlr(0.3));
	s_1->param_recur([&](Param& pmt) {
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
			epoch_cost += s_1->y_grad.abs_1d().sum_1d() / x_1.size();
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
			std::cout << "TEST COST: " << get_tsy_1(deviant_input).sum_1d() << std::endl;

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
	Sync s_0 = new sync(pseudo::tnn({ 2, 5, 1 }, pseudo::nlr(0.3)));
	s_0->param_recur([&](Param& pmt) {
		pmt = new param_mom(urd(re), learn_rate, 0, 0, beta);
		pv_0.push_back((param_mom*)pmt.get());
	});
#pragma endregion
#pragma region ORDER 1 INSTANTIATE
	const int x_1_height = 50;
	tensor x_1 = tensor::new_2d(x_1_height, 1, urd, re);
	vector<param_mom*> pv_1;
	Sequential s_1 = pseudo::tnn({ 1, pv_0.size(), pv_0.size() }, pseudo::nlr(0.3));
	s_1->param_recur([&](Param& pmt) {
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
			epoch_cost += s_1->y_grad.abs_1d().sum_1d() / x_1.size();
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
			prev_test_cost = get_tsx_1(s_1->fwd(deviant_x)).abs_1d().sum_1d();
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
	auto pmt_init = [&](Param& pmt) {
		pmt = new param(pmt_urd(re));
		pv.push_back((param*)pmt.get());
	};
#pragma endregion
#pragma region MODEL

	const size_t lstm_units = 20;
	Sync sync_in = new sync(pseudo::tnn({ 1, lstm_units }, pseudo::nlr(0.3)));
	ptr<lstm> lstm_mid = new lstm(lstm_units);
	Sync sync_out = new sync(pseudo::tnn({ lstm_units, 2 }, { pseudo::nlr(0.3), pseudo::nsm() }));
	sequential s = { sync_in.get(), lstm_mid.get(), sync_out.get() };
	s.param_recur(pmt_init);

	sync_in->prep(episode_size);
	lstm_mid->prep(episode_size);
	sync_out->prep(episode_size);
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
	auto pmt_init = [&](Param& pmt) {
		pmt = new param_mom(0, 0, 0, 0, 0.9);
		pv.push_back((param_mom*)pmt.get());
	};

	const size_t lstm_units = 25;

	Sync s_in = new sync(pseudo::tnn({ 1, lstm_units }, pseudo::nlr(0.3)));
	ptr<lstm> l_1 = new lstm(lstm_units);
	ptr<lstm> l_2 = new lstm(lstm_units);
	ptr<lstm> l_3 = new lstm(lstm_units);
	Sync s_out = new sync(pseudo::tnn({ lstm_units, 1 }, pseudo::nlr(0.3)));
	sequential s = { s_in.get(), l_1.get(), l_2.get(), l_3.get(), s_out.get() };
	s.param_recur(pmt_init);

	double state_structure = 0.001 / (double)pv.size();
	double learn_rate_structure = 0.02 / (double)pv.size();

	uniform_real_distribution<double> pmt_urd(-state_structure, state_structure);
	for (param_mom* pmt : pv) {
		pmt->state() = pmt_urd(re);
		pmt->learn_rate() = learn_rate_structure;
	}

	s_in->prep(order_1_set_len);
	l_1->prep(order_1_set_len);
	l_2->prep(order_1_set_len);
	l_3->prep(order_1_set_len);
	s_out->prep(order_1_set_len);
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
	Sequential s = pseudo::tnn({ 2, 5, 1 }, pseudo::nlr(0.3));
	s->param_recur([&](Param& pmt) {
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
	auto pmt_init = [&](Param& pmt) {
		pmt = new param_mom(urd(re), 0.0002, 0, 0, 0.9);
		pv.push_back((param_mom*)pmt.get());
	};

	const int LSTM_UNITS = 35;

	Sync s_in = new sync(pseudo::tnn({ 1, LSTM_UNITS }, pseudo::nlr(0.3)));
	ptr<lstm> l_0 = new lstm(LSTM_UNITS);
	ptr<lstm> l_1 = new lstm(LSTM_UNITS);
	Sync s_out = new sync(pseudo::tnn({ LSTM_UNITS, 1 }, pseudo::nlr(0.3)));
	sequential seq { s_in.get(), l_0.get(), l_1.get(), s_out.get() };
	seq.param_recur(pmt_init);

	s_in->prep(x.width());
	l_0->prep(x.width());
	l_1->prep(x.width());
	s_out->prep(x.width());
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

	auto pmt_init = [&](Param& pmt) {
		pmt = new param(urd(re));
		pv.push_back(pmt.get());
	};

	Model s = pseudo::tnn({ 2, 5, 1 }, pseudo::nlr(0.3));
	s->param_recur(pmt_init);
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
	auto pmt_0_init = [&](Param& pmt) {
		pmt = new param(pmt_nd(re));
		pv_0.push_back(pmt.get());
	};

	vector<param*> pv_1;
	auto pmt_1_init = [&](Param& pmt) {
		pmt = new param(pmt_nd(re));
		pv_1.push_back(pmt.get());
	};

	Model m_0 = pseudo::tnn({ 1, 10, 1 }, pseudo::nlr(0.3));
	m_0->param_recur(pmt_0_init);

	Model m_1 = pseudo::tnn({ 2, 64, pv_0.size() }, pseudo::nth());
	m_1->param_recur(pmt_1_init);

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

	auto pmt_init = [&](Param& pmt) {
		pmt = new param(pmt_urd(aurora::static_vals::random_engine));
		pv.push_back(pmt.get());
	};

	Model m = pseudo::tnn({ 2, 5, 2 }, pseudo::nlr(0.3));
	m->param_recur(pmt_init);
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
			double x = pmt_x_0(aurora::static_vals::random_engine);
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

	auto pmt_init = [&](Param& pmt) {
		pmt = new param_mom(pmt_urd(re), 0.002, 0, 0, 0.9);
		pv.push_back((param_mom*)pmt.get());
	};

	Sequential compressor = pseudo::tnn({ 54, 30, 5, 30, 54 }, pseudo::nlr(0.3));
	compressor->param_recur(pmt_init);

	Model second_half = new sequential(vector<Model>(compressor->models.begin() + 4, compressor->models.end()));
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
				compressor->fwd(test_x[ts]);
				compressor->signal(test_x[ts]);
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
	auto pmt_init = [&](Param& pmt) {
		pmt = new param_mom(pmt_urd(aurora::static_vals::random_engine), 0.0002, 0, 0, 0.9);
		pv.push_back((param_mom*)pmt.get());
	};
	
	ptr<cnl> c1 = new cnl(X_HEIGHT, X_WIDTH, 2, 2, 1);
	ptr<layer> l1 = new layer(c1->y_strides(), new layer(c1->x_strides(), pseudo::nth()));
	ptr<cnl> c2 = new cnl(c1->y_strides(), c1->x_strides(), 2, 2, 1);
	ptr<layer> l2 = new layer(c2->y_strides(), new layer(c2->x_strides(), pseudo::nth()));
	ptr<cnl> c3 = new cnl(c2->y_strides(), c2->x_strides(), 2, 2, 1);

	Sequential s = new sequential {
		c1.get(),
		l1.get(),
		c2.get(),
		l2.get(),
		c3.get(),
	};
	s->param_recur(pmt_init);

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

	const int CHECKPOINT_INTERVAL = 10000;

	for (int epoch = 0; true; epoch++) {

		double cost = 0;
		for (int ts = 0; ts < x.size(); ts++) {
			s->cycle(x[ts], y[ts]);
			cost += s->y_grad.abs_2d().sum_2d().sum_1d();
			/*if(epoch % CHECKPOINT_INTERVAL == 0)
				std::cout << s->y.to_string() << std::endl;*/
		}

		if (epoch % CHECKPOINT_INTERVAL == 0)
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
	auto pmt_init = [&](Param& pmt) {
		pmt = new param_mom(pmt_urd(re), 0.0002 / NUM_TRAINING_SETS, 0, 0, 0.9);
		pv.push_back((param_mom*)pmt.get());
	};

	Model m = pseudo::tnn({ X_LEN, X_LEN / 2, H_LEN, X_LEN / 2, X_LEN }, pseudo::nlr(0.3));
	m->param_recur(pmt_init);
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

void att_lstm_ts_test() {

	tensor ht0 = { 0, 1 };
	//tensor ht1 = { 1 };
	tensor x = {
		{0, 1},
		{1, 1},
		{0, 1},
		{1, 1},
	};
	tensor y0 = {
		0, 1
	};
	/*tensor y1 = {
		1, 1
	};*/

	uniform_real_distribution<double> pmt_urd(-1, 1);

	vector<param_mom*> pv;
	auto pmt_init = PARAM_INIT(param_mom(pmt_urd(static_vals::random_engine), 0.02, 0, 0, 0.9), pv);

	ptr<att_lstm_ts> a = new att_lstm_ts(2, { 3 });
	a->param_recur(pmt_init);
	a->prep(4);
	a->compile();
	a->unroll(4);

	for (int epoch = 0; true; epoch++) {
		double cost = 0;
		//TS 0
		a->htx.pop(ht0);
		a->cycle(x, y0);
		ht0.sub_1d(a->htx_grad, ht0);
		cost += a->y_grad.abs_1d().sum_1d();
		//TS 1
		/*a->htx.pop(ht1);
		a->cycle(x, y1);
		cost += a->y_grad.abs_1d().sum_1d();
		ht1.sub_1d(a->htx_grad, ht1);*/

		for (param_mom* pmt : pv)
			pmt->update();

		if (epoch % 10000 == 0)
			std::cout << cost << std::endl;
	}

}

void att_lstm_test() {
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

	auto pmt_init = [&](Param& pmt) {
		pmt = new param_sgd(urd(re), 0.02, 0);
		pv.push_back((param_sgd*)pmt.get());
	};

	sync* s0 = new sync(pseudo::tnn({ 2, lstm_units }, pseudo::nlr(0.3)));
	att_lstm* l = new att_lstm(lstm_units, {lstm_units});
	sync* s1 = new sync(pseudo::tnn({ lstm_units, 1 }, pseudo::nlr(0.3)));

	Sync s0_ptr = s0;
	ptr<att_lstm> l0_ptr = l;
	Sync s1_ptr = s1;

	sequential s = sequential({ s0, l, s1 });
	s.param_recur(pmt_init);

	s0->prep(4);
	l->prep(4, 4); // OUTPUT LENGTH, THEN INPUT LENGTH
	s1->prep(4);

	s.compile();

	s0->unroll(4);
	l->unroll(4, 4);
	s1->unroll(4);

	const int checkpoint_interval = 10000;

	for (int epoch = 0; true; epoch++) {
		s.cycle(x0, y0);
		if (epoch % checkpoint_interval == 0)
			std::cout << "S0: " << s.y.to_string() << std::endl;
		s.cycle(x1, y1);
		if (epoch % checkpoint_interval == 0)
			std::cout << "S1: " << s.y.to_string() << std::endl;
		for (param_sgd* pmt : pv)
			pmt->update();
	}
}

void dio_test() {

	tensor v = { 3, 7, 4, 2, 1 };

	auto D = [](double x, double a, double b) {
		return exp(-0.5 * pow(a * (x - b), 2));
	};

	auto D_deriv = [&](double x, double a, double b) {
		return a * (a * (x - b)) * D(x, a, b);
	};

	const double desired = 2;
	const double stretch = 1;
	double predicted = 0;
	
	auto get_index_grad = [&](double a, double b) {
		predicted = 0;
		for (int i = 0; i < v.size(); i++)
			predicted += v[i] * D(stretch*i, a, b);

		double y_grad = predicted - desired;

		double result = 0;
		for (int i = 0; i < v.size(); i++)
			result += y_grad * v[i] * D_deriv(stretch * i, a, b);
		return result;

	};

	double index = 0;
	double precision = 1;
	double learn_rate = 0.01;

	for (int epoch = 0; true; epoch++) {
		double index_grad = get_index_grad(precision, index);
		index -= learn_rate * index_grad;
		precision += 0.0000001;
		if (epoch % 10000 == 0) {
			std::cout << "INDEX: " << index << std::endl;
			std::cout << "PREDICT: " << predicted << std::endl;
			std::cout << "PRECISION: " << precision << std::endl;
		}
	}

}

void cos_sim_test() {
	
	size_t units = 4;

	ptr<cos_sim> p = new cos_sim(units);
	p->compile();

	uniform_real_distribution<double> urd(-100, 100);
	p->x.pop(tensor::new_2d(2, units, urd, aurora::static_vals::random_engine));

	tensor desired = 1;

	for (int epoch = 0; true; epoch++) {

		p->cycle(p->x, desired);

		tensor update = p->x_grad.mul_2d(tensor::new_2d(2, units, 2));

		p->x.sub_2d(update, p->x);

		if (epoch % 1000 == 0) {
			std::cout << p->x.to_string() << std::endl << std::endl;
			std::cout << p->y.to_string() << std::endl;
		}

	}

}

void ntm_sparsify_test() {

	size_t memory_height = 5;

	ptr<ntm_sparsify> p = new ntm_sparsify(memory_height);
	p->compile();


	uniform_real_distribution<double> urd(0, 1);

	p->x.pop(tensor::new_1d(memory_height, urd, aurora::static_vals::random_engine));
		
	double beta_des = 3;

	tensor x = tensor::new_1d(memory_height, urd, aurora::static_vals::random_engine);
	tensor y = tensor::new_1d(memory_height);
	for (int i = 0; i < x.size(); i++)
		y[i].val() = exp(beta_des * x[i]);

	for (int epoch = 0; true; epoch++) {

		p->cycle(x, y);

		p->beta[0].val() -= 0.0002 * p->beta_grad[0];

		// WE ARE SLEEPING HERE
		Sleep(10);

		if (epoch % 10 == 0) {
			std::cout << y.to_string() << std::endl << p->y.to_string() << std::endl;
			std::cout << p->beta.to_string() << std::endl << std::endl;
		}

	}

}

void normalize_test() {

	size_t units = 5;

	ptr<normalize> p = new normalize(units);
	p->compile();

	uniform_real_distribution<double> urd(-10, 10);

	p->x.pop(tensor::new_1d(units, urd, aurora::static_vals::random_engine));
	tensor y = { 0.5, 0.1, 0.1, 0.1, 0.2 };

	tensor lr_tensor = tensor::new_1d(units, 0.02);

	for (int epoch = 0; true; epoch++) {

		p->cycle(p->x, y);

		tensor update = p->x_grad.mul_1d(lr_tensor);

		p->x.sub_1d(update, p->x);

		if (epoch % 10000 == 0) {
			std::cout << y.to_string() << std::endl << p->y.to_string() << std::endl << std::endl;
		}

	}
}

void ntm_content_addresser_test() {

	size_t memory_height = 5;
	size_t memory_width = 10;

	ptr<ntm_content_addresser> p = new ntm_content_addresser(memory_height, memory_width);
	p->compile();

	uniform_real_distribution<double> urd(-1, 1);

	p->key.pop(tensor::new_1d(memory_width, urd, aurora::static_vals::random_engine));
	p->beta.pop(tensor::new_1d(1, urd, aurora::static_vals::random_engine));
	p->x.pop(tensor::new_2d(memory_height, memory_width, urd, aurora::static_vals::random_engine));

	tensor y = tensor::new_1d(memory_height);

	const size_t index_to_see = 1;
	y[index_to_see].val() = 1;

	const double lr = 0.02;
	tensor beta_lr_tensor = tensor::new_1d(1, lr);
	tensor key_lr_tensor = tensor::new_1d(memory_width, lr);
	//tensor x_lr_tensor = tensor::new_2d(memory_height, memory_width, lr);

	for (int epoch = 0; true; epoch++) {

		p->cycle(p->x, y);

		tensor beta_update = p->beta_grad.mul_1d(beta_lr_tensor);
		tensor key_update = p->key_grad.mul_1d(key_lr_tensor);
		//tensor x_update = p->x_grad.mul_2d(x_lr_tensor);

		p->beta.sub_1d(beta_update, p->beta);
		p->key.sub_1d(key_update, p->key);
		//p->x.sub_2d(x_update, p->x);

		if (epoch % 1000 == 0) {
			std::cout <<
				"INDEX: " << std::to_string(index_to_see) << std::endl <<
				p->x[index_to_see].to_string() << std::endl <<
				p->key.to_string() << std::endl <<
				p->y.to_string() << std::endl <<
				p->beta[0].to_string() << std::endl <<
				p->key.cos_sim(p->x[index_to_see]) << std::endl <<
				std::endl;
		}

	}

}

void interpolate_test() {

	size_t units = 5;

	ptr<layer> l_softmax = new layer(1, new sigmoid());
	l_softmax->compile();
	ptr<interpolate> l_interpolate = new interpolate(units);
	l_interpolate->compile();

	l_softmax->y.group_join(l_interpolate->amount);
	l_softmax->y_grad.group_join(l_interpolate->amount_grad);

	uniform_real_distribution<double> urd(-10, 10);

	const double amount_des = 0.7;

	tensor x = tensor::new_2d(2, units, urd, aurora::static_vals::random_engine);
	tensor y = tensor::new_1d(units);
	for (int i = 0; i < units; i++) {
		y[i].val() = amount_des * x[0][i] + (1 - amount_des) * x[1][i];
	}

	tensor lr_tensor = tensor::new_1d(1, 0.0002);

	for (int epoch = 0; true; epoch++) {

		l_softmax->fwd();
		l_interpolate->cycle(x, y);
		l_softmax->bwd();

		tensor update = l_softmax->x_grad.mul_1d(lr_tensor);

		l_softmax->x.sub_1d(update, l_softmax->x);

		Sleep(10);

		if (epoch % 100 == 0)
			std::cout << l_interpolate->amount.to_string() << std::endl;

	}

}

int positive_modulo(int i, int n) {
	return (i % n + n) % n;
}

void shift_test() {

	size_t units = 5;
	vector<int> valid_shifts = { -1, 0, 1 };

	ptr<layer> l_softmax = new layer(valid_shifts.size(), new sigmoid());
	l_softmax->compile();
	ptr<shift> l_shift = new shift(5, valid_shifts);
	l_shift->compile();

	l_softmax->y.group_join(l_shift->amount);
	l_softmax->y_grad.group_join(l_shift->amount_grad);

	uniform_real_distribution<double> urd(0, 1);

	tensor amount_des = { 0.3, 0.5, 0.9 };

	tensor x = tensor::new_1d(units, urd, aurora::static_vals::random_engine);
	tensor y = tensor::new_1d(units);
	for (int i = 0; i < units; i++)
		for (int j = 0; j < valid_shifts.size(); j++) {
			int dst = positive_modulo(i + valid_shifts[j], units);
			y[dst].val() += x[i] * amount_des[j];
		}

	tensor lr_tensor = tensor::new_1d(valid_shifts.size(), 0.02);

	for (int epoch = 0; true; epoch++) {

		l_softmax->fwd();
		l_shift->cycle(x, y);
		l_softmax->bwd();

		tensor update = l_softmax->x_grad.mul_1d(lr_tensor);
		l_softmax->x.sub_1d(update, l_softmax->x);

		if (epoch % 1000 == 0)
			std::cout << l_shift->amount.to_string() << std::endl;

	}

}

void power_test() {

	size_t units = 5;

	ptr<power> p = new power(units);
	p->compile();

	uniform_real_distribution<double> urd(0, 1);

	const double amount_des = 3;

	tensor x = tensor::new_1d(units, urd, aurora::static_vals::random_engine);
	tensor y = tensor::new_1d(units);
	for (int i = 0; i < units; i++)
		y[i].val() = pow(x[i], amount_des);

	tensor lr_tensor = tensor::new_1d(1, 0.02);

	for (int epoch = 0; true; epoch++) {

		p->cycle(x, y);

		tensor update = p->amount_grad.mul_1d(lr_tensor);

		p->amount.sub_1d(update, p->amount);

		if (epoch % 10000 == 0)
			std::cout << p->amount.to_string() << std::endl;

	}

}

void ntm_location_addresser_test() {
	
	size_t memory_height = 10;
	vector<int> valid_shifts = { -1, 0, 1 };

	ptr<layer> l_g_sm = new layer(1, new sigmoid());
	l_g_sm->compile();
	ptr<layer> l_s_sm = new layer(valid_shifts.size(), new sigmoid());
	l_s_sm->compile();
	ptr<ntm_location_addresser> p = new ntm_location_addresser(memory_height, valid_shifts);
	p->compile();

	l_g_sm->y.group_join(p->g);
	l_g_sm->y_grad.group_join(p->g_grad);

	l_s_sm->y.group_join(p->s);
	l_s_sm->y_grad.group_join(p->s_grad);

	uniform_real_distribution<double> urd(0, 1);

	p->g.pop(tensor::new_1d(1, urd, aurora::static_vals::random_engine));
	p->s.pop(tensor::new_1d(valid_shifts.size(), urd, aurora::static_vals::random_engine));
	p->gamma.pop(tensor::new_1d(1, urd, aurora::static_vals::random_engine));

	tensor x = tensor::new_1d(memory_height, 0.1);
	x[2].val() = 0.8;
	tensor y = tensor::new_1d(memory_height);
	y[1].val() = 1;

	const double lr = 0.2;

	tensor g_lr = tensor::new_1d(1, lr);
	tensor s_lr = tensor::new_1d(valid_shifts.size(), lr);
	tensor gamma_lr = tensor::new_1d(1, lr);
	//tensor x_lr = tensor::new_1d(memory_height, lr);

	for (int epoch = 0; true; epoch++) {

		l_g_sm->fwd();
		l_s_sm->fwd();
		p->cycle(x, y);
		l_s_sm->bwd();
		l_g_sm->bwd();

		tensor g_update = l_g_sm->x_grad.mul_1d(g_lr);
		tensor s_update = l_s_sm->x_grad.mul_1d(s_lr);
		tensor gamma_update = p->gamma_grad.mul_1d(gamma_lr);
		//tensor x_update = p->x_grad.mul_1d(x_lr);

		l_g_sm->x.sub_1d(g_update, l_g_sm->x);
		l_s_sm->x.sub_1d(s_update, l_s_sm->x);
		p->gamma.sub_1d(gamma_update, p->gamma);
		//p->x.sub_1d(x_update, p->x);

		if (epoch % 1000 == 0)
			std::cout << y.to_string() << std::endl << p->y.to_string() 
			<< std::endl << std::endl;

	}

}

void ntm_addresser_test() {

	size_t memory_height = 5;
	size_t memory_width = 5;
	vector<int> valid_shifts = { -1, 0, 1 };

	ptr<layer> l_g_sm = new layer(1, new sigmoid());
	l_g_sm->compile();
	ptr<layer> l_s_sm = new layer(valid_shifts.size(), new sigmoid());
	l_s_sm->compile();
	ptr<ntm_addresser> p = new ntm_addresser(memory_height, memory_width, valid_shifts);
	p->compile();

	uniform_real_distribution<double> urd(-1, 1);
	uniform_real_distribution<double> sm_urd(0, 1);
	uniform_real_distribution<double> pos_urd(1, 3);

	const size_t prev_selected_index = 3;

	p->wx.pop(tensor::new_1d(memory_height));
	p->wx[prev_selected_index].val() = 1;

	p->x.pop(tensor::new_2d(memory_height, memory_width, urd, aurora::static_vals::random_engine));
	p->key.pop(tensor::new_1d(memory_width, urd, aurora::static_vals::random_engine));
	p->beta.pop(tensor::new_1d(1, urd, aurora::static_vals::random_engine));
	p->gamma.pop(tensor::new_1d(1, pos_urd, aurora::static_vals::random_engine));
	l_g_sm->x.pop(tensor::new_1d(1, sm_urd, aurora::static_vals::random_engine));
	l_s_sm->x.pop(tensor::new_1d(valid_shifts.size(), sm_urd, aurora::static_vals::random_engine));

	l_g_sm->y.group_join(p->g);
	l_g_sm->y_grad.group_join(p->g_grad);
	l_s_sm->y.group_join(p->s);
	l_s_sm->y_grad.group_join(p->s_grad);

	const size_t selected_index = 1;
	tensor y = tensor::new_1d(memory_height);
	y[selected_index].val() = 1;

	const double lr = 0.002;

	//tensor wx_lr = tensor::new_1d(memory_height, lr);
	tensor x_lr = tensor::new_2d(memory_height, memory_width, lr);
	tensor key_lr = tensor::new_1d(memory_width, lr);
	tensor beta_lr = tensor::new_1d(1, lr);
	tensor gamma_lr = tensor::new_1d(1, lr);
	tensor g_lr = tensor::new_1d(1, lr);
	tensor s_lr = tensor::new_1d(valid_shifts.size(), lr);

	for (int epoch = 0; true; epoch++) {

		l_g_sm->fwd();
		l_s_sm->fwd();
		p->cycle(p->x, y);
		l_s_sm->bwd();
		l_g_sm->bwd();

		// NEVER UPDATE WX LIKE THIS. WX SHOULD BE BETWEEN 0 AND 1
		//tensor wx_update = p->wx_grad.mul_1d(wx_lr);
		tensor x_update = p->x_grad.mul_2d(x_lr);
		tensor key_update = p->key_grad.mul_1d(key_lr);
		tensor beta_update = p->beta_grad.mul_1d(beta_lr);
		tensor gamma_update = p->gamma_grad.mul_1d(gamma_lr);
		tensor g_update = l_g_sm->x_grad.mul_1d(g_lr);
		tensor s_update = l_s_sm->x_grad.mul_1d(s_lr);
		
		//p->wx.sub_1d(wx_update, p->wx);
		p->x.sub_2d(x_update, p->x);
		p->key.sub_1d(key_update, p->key);
		p->beta.sub_1d(beta_update, p->beta);
		p->gamma.sub_1d(gamma_update, p->gamma);
		l_g_sm->x.sub_1d(g_update, l_g_sm->x);
		l_s_sm->x.sub_1d(s_update, l_s_sm->x);

		if (epoch % 1000 == 0)
			std::cout << y.to_string() << std::endl << p->y.to_string()
			<< std::endl << std::endl;

	}

}

void ntm_rh_test() {

	tensor x_0 = { 0, 1, 2, 3, 4 };
	tensor des_k_0 = { 0, 1, 2, 3, 4 };
	tensor des_beta_0 = { 3 };
	tensor des_gamma_0 = { 0.75 };
	tensor des_g_0 = { 1 };
	tensor des_s_0 = { 0.1, 0.5, 0.1 };

	tensor x_1 = { 1, 2, 3, 4, 5 };
	tensor des_k_1 = { 5, 4, 3, 2, 1 };
	tensor des_beta_1 = { 9 };
	tensor des_gamma_1 = { 0.99 };
	tensor des_g_1 = { 0 };
	tensor des_s_1 = { 0.5, 0.2, 0.9 };

	vector<param_sgd*> pv;
	uniform_real_distribution<double> urd(-1, 1);

	ntm_rh nrh = ntm_rh(5, { 6, 7 }, 3);
	nrh.param_recur(PARAM_INIT(param_sgd(urd(aurora::static_vals::random_engine), 0.002, 0), pv));
	nrh.compile();

	for (int epoch = 0; epoch < 1000000; epoch++) {

		double cost = 0;

			nrh.fwd(x_0);

			nrh.key_grad.pop(nrh.key.sub_1d(des_k_0));
			nrh.beta_grad.pop(nrh.beta.sub_1d(des_beta_0));
			nrh.gamma_grad.pop(nrh.gamma.sub_1d(des_gamma_0));
			nrh.g_grad.pop(nrh.g.sub_1d(des_g_0));
			nrh.s_grad.pop(nrh.s.sub_1d(des_s_0));

			nrh.bwd();

			cost +=
				nrh.key_grad.abs_1d().sum_1d() +
				nrh.beta_grad.abs_1d().sum_1d() +
				nrh.gamma_grad.abs_1d().sum_1d() +
				nrh.g_grad.abs_1d().sum_1d() +
				nrh.s_grad.abs_1d().sum_1d();

			nrh.fwd(x_1);

			nrh.key_grad.pop(nrh.key.sub_1d(des_k_1));
			nrh.beta_grad.pop(nrh.beta.sub_1d(des_beta_1));
			nrh.gamma_grad.pop(nrh.gamma.sub_1d(des_gamma_1));
			nrh.g_grad.pop(nrh.g.sub_1d(des_g_1));
			nrh.s_grad.pop(nrh.s.sub_1d(des_s_1));

			nrh.bwd();

			cost +=
				nrh.key_grad.abs_1d().sum_1d() +
				nrh.beta_grad.abs_1d().sum_1d() +
				nrh.gamma_grad.abs_1d().sum_1d() +
				nrh.g_grad.abs_1d().sum_1d() +
				nrh.s_grad.abs_1d().sum_1d();


		for (param_sgd* pmt : pv)
			pmt->update();


		if (epoch % 10000 == 0) {
			std::cout << cost << std::endl;
		}

	}

}

void ntm_wh_test() {

	tensor x_0 = { 0, 1, 2, 3, 4 };
	tensor des_k_0 = { 1, 2, 3, 4, 5 };
	tensor des_beta_0 = { 2 };
	tensor des_gamma_0 = { 3 };
	tensor des_g_0 = { 0.5 };
	tensor des_s_0 = { 0.1, 0.2, 0.3 };
	tensor des_e_0 = { 0, 0.1, 0.9, 0.9, 0.2 };
	tensor des_a_0 = { 1, 2, 3, 4, 5 };

	tensor x_1 = { 1, 2, 3, 3, 4 };
	tensor des_k_1 = { 0, 1, 2, 3, 4 };
	tensor des_beta_1 = { 10 };
	tensor des_gamma_1 = { 5 };
	tensor des_g_1 = { 0.2 };
	tensor des_s_1 = { 0.3, 0.9, 0.1 };
	tensor des_e_1 = { 0.3, 0.2, 0.1, 0.2, 0.3 };
	tensor des_a_1 = { 3, 0, 0, 1, 2 };

	vector<param_sgd*> pv;
	uniform_real_distribution<double> urd(-1, 1);

	ntm_wh nwh = ntm_wh(5, { 6, 7 }, 3);
	nwh.param_recur(PARAM_INIT(param_sgd(urd(aurora::static_vals::random_engine), 0.0002, 0), pv));
	nwh.compile();

	for (int epoch = 0; epoch < 1000000; epoch++) {

		double cost = 0;

		nwh.fwd(x_0);

		nwh.internal_rh->key_grad.pop(nwh.internal_rh->key.sub_1d(des_k_0));
		nwh.internal_rh->beta_grad.pop(nwh.internal_rh->beta.sub_1d(des_beta_0));
		nwh.internal_rh->gamma_grad.pop(nwh.internal_rh->gamma.sub_1d(des_gamma_0));
		nwh.internal_rh->g_grad.pop(nwh.internal_rh->g.sub_1d(des_g_0));
		nwh.internal_rh->s_grad.pop(nwh.internal_rh->s.sub_1d(des_s_0));
		nwh.e_grad.pop(nwh.e.sub_1d(des_e_0));
		nwh.a_grad.pop(nwh.a.sub_1d(des_a_0));

		nwh.bwd();

		cost +=
			nwh.internal_rh->key_grad.abs_1d().sum_1d() +
			nwh.internal_rh->beta_grad.abs_1d().sum_1d() +
			nwh.internal_rh->gamma_grad.abs_1d().sum_1d() +
			nwh.internal_rh->g_grad.abs_1d().sum_1d() +
			nwh.internal_rh->s_grad.abs_1d().sum_1d() +
			nwh.e_grad.abs_1d().sum_1d() +
			nwh.a_grad.abs_1d().sum_1d();

		nwh.fwd(x_1);

		nwh.internal_rh->key_grad.pop(nwh.internal_rh->key.sub_1d(des_k_1));
		nwh.internal_rh->beta_grad.pop(nwh.internal_rh->beta.sub_1d(des_beta_1));
		nwh.internal_rh->gamma_grad.pop(nwh.internal_rh->gamma.sub_1d(des_gamma_1));
		nwh.internal_rh->g_grad.pop(nwh.internal_rh->g.sub_1d(des_g_1));
		nwh.internal_rh->s_grad.pop(nwh.internal_rh->s.sub_1d(des_s_1));
		nwh.e_grad.pop(nwh.e.sub_1d(des_e_0));
		nwh.a_grad.pop(nwh.a.sub_1d(des_a_0));

		nwh.bwd();

		cost +=
			nwh.internal_rh->key_grad.abs_1d().sum_1d() +
			nwh.internal_rh->beta_grad.abs_1d().sum_1d() +
			nwh.internal_rh->gamma_grad.abs_1d().sum_1d() +
			nwh.internal_rh->g_grad.abs_1d().sum_1d() +
			nwh.internal_rh->s_grad.abs_1d().sum_1d() +
			nwh.e_grad.abs_1d().sum_1d() +
			nwh.a_grad.abs_1d().sum_1d();;

		for (param_sgd* pmt : pv)
			pmt->update();

		if (epoch % 1000 == 0) {
			std::cout << cost << std::endl;
		}

	}

}

void ntm_reader_test() {

	size_t memory_height = 5;
	size_t memory_width = 5;
	vector<int> valid_shifts = { -1, 0, 1 };

	uniform_real_distribution<double> pmt_urd(-0.1, 0.1);
	uniform_real_distribution<double> mem_urd(-10, 10);

	vector<param_sgd*> pv;
	auto pmt_init = PARAM_INIT(
		param_sgd(pmt_urd(aurora::static_vals::random_engine), 0.002, 0), pv);

	Ntm_reader p = new ntm_reader(memory_height, memory_width, valid_shifts, {memory_width, memory_width + 5});
	p->param_recur(pmt_init);
	p->compile();

	p->mx.pop(tensor::new_2d(memory_height, memory_width, mem_urd, aurora::static_vals::random_engine));
	//p->wx[1].val() = 1;

	const size_t selected_index = 1;
	tensor y = p->mx[selected_index];

	/*tensor y = tensor::new_1d(memory_width);
	for (int i = 0; i < memory_width; i++)
		y[i].val() = 0.5*p->mx[2][i] + 0.5*p->mx[4][i];*/

	for (int epoch = 0; true; epoch++) {

		p->cycle(p->x, y);

		for (param_sgd* pmt : pv)
			pmt->update();

		if (epoch % 1000 == 0)
			std::cout <<
			"WY: " << p->internal_addresser->wy.to_string() << std::endl <<
			"DESIRED: " << y.to_string() << std::endl <<
			"ACTUAL:  " << p->y.to_string() << std::endl <<
			"GAMMA: " << p->internal_addresser->gamma[0].to_string() << std::endl << std::endl;

	}

}

void ntm_writer_test() {

	size_t memory_height = 5;
	size_t memory_width = 5;
	vector<int> valid_shifts = { -1, 0, 1 };

	uniform_real_distribution<double> pmt_urd(-0.1, 0.1);
	uniform_real_distribution<double> mem_urd(-10, 10);

	default_random_engine dre(27);

	vector<param_mom*> pv;
	auto pmt_init = PARAM_INIT(
		param_mom(pmt_urd(dre), 0.00002, 0, 0, 0.9), pv);

	ptr<ntm_writer> p = new ntm_writer(memory_height, memory_width, valid_shifts, { memory_width });
	p->param_recur(pmt_init);
	p->compile();

	p->mx.pop(tensor::new_2d(memory_height, memory_width, mem_urd, aurora::static_vals::random_engine));
	p->wx[4].val() = 0.4;
	p->wx[3].val() = 0.6;

	const size_t selected_index = 1;
	tensor y = p->mx.clone();
	y[selected_index].add_1d(tensor::new_1d(memory_width, 1), y[selected_index]);

	/*tensor y = tensor::new_1d(memory_width);
	for (int i = 0; i < memory_width; i++)
		y[i].val() = 0.5*p->mx[2][i] + 0.5*p->mx[4][i];*/

	for (int epoch = 0; true; epoch++) {

		p->cycle(p->x, y);

		for (param_mom* pmt : pv)
			pmt->update();

		double cost = p->y_grad.abs_2d().sum_2d().sum_1d();

		if (epoch % 10000 == 0)
			std::cout << std::to_string(cost) << std::endl <<
			"G: " << p->internal_addresser->g[0].to_string() << std::endl <<
			"S: " << p->internal_addresser->s.to_string() << std::endl <<
			"GAMMA: " << p->internal_addresser->gamma[0].to_string() << std::endl <<
			"BETA:     " << p->internal_addresser->beta[0].to_string() << std::endl <<
			"WY: " << p->wy.to_string() << std::endl <<
			"SIM_TENSOR: " << p->internal_addresser->internal_content_addresser->internal_similarity->y.to_string() << std::endl <<
			"A: " << p->internal_head->a.to_string() << std::endl <<
			"DES Y: " << y[selected_index].to_string() << std::endl <<
			"ACT Y: " << p->y[selected_index].to_string() << std::endl <<
			std::endl << std::endl;
	}

}

void ntm_test() {

	size_t memory_height = 5;
	size_t memory_width = 5;
	size_t num_readers = 1;
	size_t num_writers = 1;
	vector<int> valid_shifts = { -1, 0, 1 };
	vector<size_t> head_hidden_dims = { memory_width };

	uniform_real_distribution<double> pmt_urd(-1, 1);
	uniform_real_distribution<double> ts_urd(-10, 10);
	uniform_real_distribution<double> mem_urd(-1, 1);
	default_random_engine dre(27);

	vector<Param> pv;

	auto pmt_init = PARAM_INIT(param_mom(pmt_urd(dre), 0.002, 0, 0, 0.9), pv);

	Sync s_in = new sync(pseudo::tnn({ 2, memory_width }, pseudo::nlr(0.3)));

	Ntm p = new ntm(
		memory_height,
		memory_width,
		num_readers,
		num_writers,
		valid_shifts,
		head_hidden_dims);

	Sync s_out = new sync(pseudo::tnn({ memory_width, 1 }, pseudo::nlr(0.3)));

	Sequential s = new sequential({ s_in, p, s_out });
	s->param_recur(pmt_init);

	const size_t num_ts = 4;

	s_in->prep(num_ts);
	p->prep(num_ts);
	s_out->prep(num_ts);
	s->compile();
	s_in->unroll(num_ts);
	p->unroll(num_ts);
	s_out->unroll(num_ts);

	tensor x = {
		{
			{0, 0},
			{0, 1},
			{1, 0},
			{1, 1}
		},
		{
			{1, 0},
			{0, 1},
			{1, 0},
			{1, 1}
		},
	};
	tensor y = {
		{
			{0},
			{1},
			{1},
			{0}
		},
		{
			{1},
			{1},
			{1},
			{1}
		},
	};

	const size_t checkpoint_interval = 1000;

	for (int epoch = 0; epoch < 100000; epoch++) {

		double cost = 0;

		for (int ts = 0; ts < x.size(); ts++) {
			s->cycle(x[ts], y[ts]);
			cost += s->y_grad.abs_2d().sum_2d().sum_1d();

			if (epoch % checkpoint_interval == 0)
				std::cout << s->y.to_string() << std::endl;
		}

		for (param* pmt : pv)
			pmt->update();

		if (epoch % checkpoint_interval == 0)
			std::cout << cost << std::endl << std::endl;
	}

}

void stacked_recurrent_test() {
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

	auto pmt_init = [&](Param& pmt) {
		pmt = new param_sgd(urd(re), 0.02, 0);
		pv.push_back(pmt);
	};
	Stacked_recurrent s = new stacked_recurrent({
		new sync(pseudo::tnn({ 2, lstm_units }, pseudo::nlr(0.3))),
		new lstm(lstm_units),
		new sync(pseudo::tnn({ lstm_units, 1 }, pseudo::nlr(0.3)))
	});
	s->param_recur(pmt_init);

	s->prep(4);
	s->compile();
	s->unroll(4);

	const int checkpoint_interval = 10000;

	for (int epoch = 0; true; epoch++) {
		s->cycle(x0, y0);
		if (epoch % checkpoint_interval == 0)
			std::cout << "S0: " << s->y.to_string() << std::endl;
		s->cycle(x1, y1);
		if (epoch % checkpoint_interval == 0)
			std::cout << "S1: " << s->y.to_string() << std::endl;
		for (param_sgd* pmt : pv) {
			pmt->state() -= pmt->learn_rate() * pmt->gradient();
			pmt->gradient() = 0;
		}
	}

}

void lstm_compiled_test() {

	param_vector pv;
	Stacked_recurrent s = basic::basic_lstm_mdim(2, 10, 1, 4, pv);
	s->unroll(4);

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

	const int checkpoint_interval = 10000;

	for (int epoch = 0; epoch < 100000; epoch++) {
		s->cycle(x0, y0);
		if (epoch % checkpoint_interval == 0)
			std::cout << "S0: " << s->y.to_string() << std::endl;
		s->cycle(x1, y1);
		if (epoch % checkpoint_interval == 0)
			std::cout << "S1: " << s->y.to_string() << std::endl;
		pv.update();
	}

}

void test_test() {

	param* p = new param();
	Param p1 = p;
	Param p2 = new param();

	test_test();


}

int main() {

	srand(time(NULL));
	
	lstm_compiled_test();

	return 0;

}