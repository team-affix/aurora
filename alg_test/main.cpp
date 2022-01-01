#pragma once
#include "aurora.h"
#include "affix-base/threading.h"
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
#include <random>
#include <csignal>
#include "affix-base/timing.h"

#define L(call) [&]{call;}

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
using std::vector;
using std::string;
using std::function;
using affix_base::threading::persistent_thread;


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

void pl_export_to_file(std::string file_name, param_vector& a_pl) {
	for (int i = 0; i < a_pl.size(); i++)
		if (isnan(a_pl[i]->state()) || isinf(a_pl[i]->state())) {
			std::cout << "ERROR: PARAM IS NAN OR INF" << std::endl;
			return;
		}
	ofstream ofs(file_name);
	for (int i = 0; i < a_pl.size(); i++)
		ofs << a_pl[i]->state() << std::endl;
	ofs.close();
}

void pl_import_from_file(std::string file_name, param_vector& a_pl) {
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
			mat_4.link(mat_5);
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
		t2[0].link(t3[0]);
		assert(t1[0] == 5);
	}

	{
		tensor t1 = { 0, 0, 0, 0 };
		tensor t2 = t1.range(0, 2);

	}

	{
		tensor t1 = tensor::new_1d(1, 1);
		tensor t2 = tensor::new_1d(1, 2);

		tensor t3 = t1.cat(t2);

		t3[0].val() = 3;
		assert(t1[0].val() == 3);

	}

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

	param_vector pv;

	uniform_real_distribution<double> urd(-1, 1);
	default_random_engine re(25);

	Sequential s = pseudo::tnn({ 2, 5, 1 }, { pseudo::nlr(0.3), pseudo::nlr(0.3), pseudo::nsm() });
	s->param_recur([&](Param& pmt) {
		pmt = new param_mom(0.02, 0.9);
		pv.push_back((param_mom*)pmt.get());
	});

	Ce_loss m = new ce_loss(s);
	m->compile();

	pv.randomize();
	pv.normalize();

	printf("");

	const size_t checkpoint_interval = 10000;

	for (int epoch = 0; epoch < 1000000; epoch++) {

		if (epoch % checkpoint_interval == 0) {
			printf("\033[%d;%dH", 0, 0);
			std::cout << epoch << std::endl;
		}

		for (int tsIndex = 0; tsIndex < x.size(); tsIndex++) {
			m->cycle(x[tsIndex], y[tsIndex]);

			if (epoch % checkpoint_interval == 0)
				std::cout << x[tsIndex].to_string() << " " << s->m_y.to_string() << std::endl;
		}

		for (param_mom* pmt : pv)
			pmt->update();
	}

	for (param_mom* pmt : pv) {
		std::cout << pmt->state() << std::endl;
	}

}

void tnn_compiled_xor_test() {
	
	using aurora::params::param_vector;

	param_vector pv;

	Model s = pseudo::tnn({ 2, 5, 1 }, pseudo::nlr(0.3));

	s->param_recur(pseudo::param_init(new param_mom(0.02, 0.9), pv));
	pv.rand_norm();

	Mse_loss m = new mse_loss(s);

	m->compile();
	
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
			m->cycle(x[tsIndex], y[tsIndex]);
			if (epoch % checkpoint_interval == 0)
				std::cout << x[tsIndex].to_string() << " " << s->m_y.to_string() << std::endl;
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

	param_vector pl;
	uniform_real_distribution<double> urd(-1, 1);
	Sequential s = pseudo::tnn({ 2, 17, 1 }, pseudo::nlr(0.3));
	s->param_recur(pseudo::param_init(new param_mom(0.0002, 0.9), pl));

	Mse_loss m = new mse_loss(s);
	m->compile();

	pl.randomize();
	pl.normalize();
	
	for (int epoch = 0; epoch < 1000000; epoch++) {

		if (epoch % 10000 == 0) {
			printf("\033[%d;%dH", 0, 0);
			std::cout << epoch << std::endl;
		}

		double cost = 0;

		for (int tsIndex = 0; tsIndex < x.size(); tsIndex++) {
			m->cycle(x[tsIndex], y[tsIndex]);
			cost += s->m_y_grad.abs_1d().sum_1d();
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
	
	param_vector pl;
	Sync s_prev = new sync(pseudo::tnn({ 2, 5, 1 }, pseudo::nlr(0.3)));
	s_prev->param_recur(pseudo::param_init(new param_sgd(0.02), pl));
	s_prev->prep(4);

	Sync s = (sync*)s_prev->clone();
	
	Mse_loss m = new mse_loss(s);
	m->compile();

	pl.randomize();
	pl.normalize();

	s->unroll(4);

	for (int epoch = 0; epoch < 1000000; epoch++) {
		m->cycle(x, y);

		if (epoch % 10000 == 0) {
			printf("\033[%d;%dH", 0, 0);
			std::cout << epoch << std::endl;
		}

		if (epoch % 10000 == 0)
			std::cout << x.to_string() << std::endl << s->m_y.to_string() << std::endl << std::endl;

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

	Mse_loss m = new mse_loss(s);
	m->compile();

	double best_cost = INFINITY;

	const int checkpoint_interval = 5;
	const int max_fail_lr_update = 35;

	int fail_index = 0;

	double checkpoint_cost = 0;

	for (int epoch = 0; true; epoch++) {
		double epoch_cost = 0;
		for (int i = 0; i < mini_batch_len; i++) {
			int ts_index = random(0, x_order_2_len);
			m->cycle(train_x[ts_index], train_x[ts_index]);
			if(epoch % checkpoint_interval == 0)
				epoch_cost += s->m_y_grad.abs_1d().sum_1d();
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

	param_vector pv;

	auto pmt_init = [&](Param& pmt) {
		pmt = new param_sgd(0.02);
		pv.push_back(pmt);
	};

	sync* s0 = new sync(pseudo::tnn({ 2, lstm_units }, pseudo::nlr(0.3)));
	lstm* l = new lstm(lstm_units);
	sync* s1 = new sync(pseudo::tnn({ lstm_units, 1 }, pseudo::nlr(0.3)));

	Sync s0_ptr = s0;
	ptr<lstm> l0_ptr = l;
	Sync s1_ptr = s1;

	Sequential s = new sequential({ s0, l, s1 });
	s->param_recur(pmt_init);

	s0->prep(4);
	l->prep(4);
	s1->prep(4);

	Mse_loss m = new mse_loss(s);
	m->compile();

	pv.randomize();
	pv.normalize();

	s0->unroll(4);
	l->unroll(4);
	s1->unroll(4);

	const int checkpoint_interval = 10000;

	for (int epoch = 0; epoch < 100000; epoch++) {
		m->cycle(x0, y0);
		if (epoch % checkpoint_interval == 0)
			std::cout << "S0: " << s->m_y.to_string() << std::endl;
		m->cycle(x1, y1);
		if (epoch % checkpoint_interval == 0)
			std::cout << "S1: " << s->m_y.to_string() << std::endl;
		for (param_sgd* pmt : pv) {
			pmt->state() -= pmt->learn_rate() * pmt->gradient();
			pmt->gradient() = 0;
		}
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

//tensor get_zil_data() {
//	string zil_csv_directory = "D:\\files\\files\\csv\\zil";
//	tensor raw;
//	for (const auto& entry : std::filesystem::directory_iterator(zil_csv_directory)) {
//		tensor csv = tensor::new_2d(num_lines(entry.path().u8string()), 2);
//		string line;
//		std::ifstream ifs(entry);
//		int row = 0;
//		while (std::getline(ifs, line)) {
//			std::istringstream s(line);
//			std::string field;
//			int col = 0;
//			while (getline(s, field, ',')) {
//				string trimmed = trim(field, "\"");
//				if (col == 1)
//					csv[row][col] = std::stod(trimmed);
//				else
//					csv[row][col] = std::stod(trimmed) / 1000 / 60;
//				col++;
//			}
//			row++;
//		}
//		raw.vec().push_back(csv);
//	}
//	return raw;
//}

void zil_mapper() {
	/*tensor raw = get_zil_data();
	tensor x;
	tensor y;
	const int num_from_each_csv = 10000;
	for (tensor& csv : raw.vec()) {
		for (int ts = 0; ts < num_from_each_csv; ts++)
		{
			int minutes_wait = rand() % 240;

		}
	}*/
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

	param_vector pv;
	auto pmt_init = [&](Param& pmt) {
		pmt = new param_mom(0, 0.9);
		pv.push_back((param_mom*)pmt.get());
	};

	const size_t lstm_units = 25;

	Sync s_in = new sync(pseudo::tnn({ 1, lstm_units }, pseudo::nlr(0.3)));
	ptr<lstm> l_1 = new lstm(lstm_units);
	ptr<lstm> l_2 = new lstm(lstm_units);
	ptr<lstm> l_3 = new lstm(lstm_units);
	Sync s_out = new sync(pseudo::tnn({ lstm_units, 1 }, pseudo::nlr(0.3)));
	Sequential s = new sequential{ s_in.get(), l_1.get(), l_2.get(), l_3.get(), s_out.get() };
	s->param_recur(pmt_init);

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

	Mse_loss m = new mse_loss(s);
	m->compile();

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
			m->cycle(train_x[ts], train_y[ts]);
			train_cost += m->m_y_grad.abs_2d().sum_2d().sum_1d() * cost_structure;
		}

		double test_cost = 0;
		for (int ts = 0; ts < order_2_set_len; ts++) {
			m->fwd(test_x[ts]);
			m->signal(test_y[ts]);
			test_cost += m->m_y_grad.abs_2d().sum_2d().sum_1d() * cost_structure;
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
	Mse_loss m = new mse_loss(s);
	m->compile();
#pragma endregion
#pragma region LINK TENSOR
	tensor pmt_link_tensor = tensor::new_1d(pv.size());
	for (int i = 0; i < pmt_link_tensor.size(); i++)
		pmt_link_tensor[i].m_val_ptr.link(pv[i]->m_state_ptr);
#pragma endregion
#pragma region GET REWARD
	auto get_reward = [&](genome& a_genome) {
		pmt_link_tensor.pop(a_genome);
		double cost = 0;
		for (int i = 0; i < x.size(); i++) {
			s->fwd(x[i]);
			m->signal(y[i]);
			cost += s->m_y_grad.abs_1d().sum_1d();
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

void test_param_rcv() {

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

	param_vector pv;

	Model s = pseudo::tnn({ 2, 5, 1 }, pseudo::nlr(0.3));

	Mse_loss m = new mse_loss(s);

	const double GAMMA = 0.002;
	const double BETA = 0.9;

	s->param_recur(pseudo::param_init(new param_rcv(GAMMA, BETA), pv));
	m->compile();

	pv.randomize();
	pv.normalize();

	std::normal_distribution<double> nd(0, 1);

	auto get_rcv = [&]() {
		return nd(re);
	};

	auto get_reward = [&]() {
		double cost = 0;
		for (int ts = 0; ts < x.size(); ts++) {
			s->fwd(x[ts]);
			m->signal(y[ts]);
			cost += s->m_y_grad.abs_1d().sum_1d();
		}
		return 1 / cost;
	};

	for (int epoch = 0; true; epoch++) {
		for (Param_rcv pmt : pv)
			pmt->update(nd(re));

		double reward = get_reward();
		double cost = 1 / reward;

		for (Param_rcv pmt : pv)
			pmt->reward(reward);

		if (epoch % 1000 == 0)
			for (Param_rcv pmt : pv)
				pmt->learn_rate() = GAMMA * std::tanh(cost);

		if (epoch % 10000 == 0) {
			std::cout << "REWARD: " << reward << ", COST: " << 1 / reward << std::endl;
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
		pmt_link[i].m_val_ptr.link(pv[i]->m_state_ptr);

	tensor order_0_x = m->m_x[0].link();
	tensor order_0_y_hat = m->m_y[0].link();
	tensor order_1_x = m->m_x[1].link();
	tensor order_1_param_index = m->m_y[0].link();
	tensor order_1_param_state = m->m_y[1].link();

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

void cnl_test()
{
	const size_t FILTER_HEIGHT = 2;
	const size_t FILTER_WIDTH = 3;
	const size_t MAX_INPUT_HEIGHT = 10;
	const size_t MAX_INPUT_WIDTH = 10;

	tensor x = tensor::new_2d(MAX_INPUT_HEIGHT, MAX_INPUT_WIDTH);
	for (int i = 0; i < x.height(); i++)
		for (int j = 0; j < x.width(); j++)
			x[i][j].val() = i * x.width() + j;

	std::cout << "X of first filter should be: " << x.range_2d(0,0,FILTER_HEIGHT,FILTER_WIDTH).to_string() << std::endl;

	uniform_real_distribution<double> l_urd(-1, 1);
	param_vector pv;

	Cnl c = new cnl(FILTER_HEIGHT, FILTER_WIDTH, 1, 2);
	c->param_recur(pseudo::param_init(new param_mom(0.00002, 0.9), pv));
	c->prep_for_input(MAX_INPUT_HEIGHT, MAX_INPUT_WIDTH);

	Mse_loss m = new mse_loss(c);

	m->compile();

	pv.randomize();
	pv.normalize();

	c->unroll_for_input(MAX_INPUT_HEIGHT, MAX_INPUT_WIDTH);

	c->m_x.pop(x);
	std::cout << "X of first filter is:        " << c->m_filters->m_unrolled[0]->m_x.roll(FILTER_WIDTH).to_string() << std::endl;

	tensor y = tensor::new_2d(c->y_strides(), c->x_strides(), 3);

	for (int epoch = 0; epoch < 10000; epoch++)
	{
		m->cycle(x, y);
		pv.update();

		if (epoch % 1000 == 0)
			std::cout << c->m_y_grad.abs_2d().sum_2d().sum_1d() << std::endl;

	}

}

void cnn_test() {

	const size_t X_HEIGHT = 10;
	const size_t X_WIDTH = 10;

	tensor x = {
		tensor::new_2d(X_HEIGHT, X_WIDTH, 1),
		tensor::new_2d(X_HEIGHT, X_WIDTH, 2),
		tensor::new_2d(X_HEIGHT, X_WIDTH, 3),
	};

	uniform_real_distribution<double> pmt_urd(-1, 1);

	param_vector pv;
	
	ptr<cnl> c1 = new cnl(2, 2, 1);
	c1->prep_for_input(X_HEIGHT, X_WIDTH);

	ptr<layer> l1 = new layer(c1->y_strides(), new layer(c1->x_strides(), pseudo::nlr(0.3)));

	ptr<cnl> c2 = new cnl(2, 2, 1);
	c2->prep_for_input(c1->y_strides(), c1->x_strides());

	ptr<layer> l2 = new layer(c2->y_strides(), new layer(c2->x_strides(), pseudo::nlr(0.3)));

	ptr<cnl> c3 = new cnl(2, 2, 1);
	c3->prep_for_input(c2->y_strides(), c2->x_strides());

	Sequential s = new sequential {
		c1,
		l1,
		c2,
		l2,
		c3,
	};
	s->param_recur(pseudo::param_init(new param_mom(0.0002, 0.9), pv));

	Mse_loss m = new mse_loss(s);

	std::cout << "COMPILING MODEL" << std::endl;
	m->compile();

	pv.randomize();
	pv.normalize();

	tensor y = {
		tensor::new_2d(c3->y_strides(), c3->x_strides(), 1),
		tensor::new_2d(c3->y_strides(), c3->x_strides(), 2),
		tensor::new_2d(c3 ->y_strides(), c3->x_strides(), 3),
	};

	c1->unroll_for_input(X_HEIGHT, X_WIDTH);
	c2->unroll_for_input(c1->y_strides(), c1->x_strides());
	c3->unroll_for_input(c2->y_strides(), c2->x_strides());

	std::cout << "TRAINING MODEL" << std::endl;

	const int CHECKPOINT_INTERVAL = 1000;

	for (int epoch = 0; true; epoch++) {

		double cost = 0;
		for (int ts = 0; ts < x.size(); ts++) {
			m->cycle(x[ts], y[ts]);
			cost += s->m_y_grad.abs_2d().sum_2d().sum_1d();
			/*if(epoch % CHECKPOINT_INTERVAL == 0)
				std::cout << s->y.to_string() << std::endl;*/
		}

		if (epoch % CHECKPOINT_INTERVAL == 0)
			std::cout << cost << std::endl;

		pv.update();

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

	param_vector pv;
	auto pmt_init = pseudo::param_init(new param_mom(0.02, 0.9), pv);

	ptr<att_lstm_ts> a = new att_lstm_ts(2, { 3 });
	a->param_recur(pmt_init);
	a->prep(4);

	Mse_loss m = new mse_loss(a);

	m->compile();

	pv.randomize();
	pv.normalize();

	a->unroll(4);

	for (int epoch = 0; true; epoch++) {
		double cost = 0;
		//TS 0
		a->m_htx.pop(ht0);
		m->cycle(x, y0);
		ht0.sub_1d(a->m_htx_grad, ht0);
		cost += a->m_y_grad.abs_1d().sum_1d();
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

	param_vector pv;

	auto pmt_init = [&](Param& pmt) {
		pmt = new param_sgd(0.02);
		pv.push_back((param_sgd*)pmt.get());
	};

	sync* s0 = new sync(pseudo::tnn({ 2, lstm_units }, pseudo::nlr(0.3)));
	att_lstm* l = new att_lstm(lstm_units, {lstm_units});
	sync* s1 = new sync(pseudo::tnn({ lstm_units, 1 }, pseudo::nlr(0.3)));

	Sync s0_ptr = s0;
	ptr<att_lstm> l0_ptr = l;
	Sync s1_ptr = s1;

	Sequential s = new sequential({ s0, l, s1 });
	s->param_recur(pmt_init);

	s0->prep(4);
	l->prep(4, 4); // OUTPUT LENGTH, THEN INPUT LENGTH
	s1->prep(4);

	Mse_loss m = new mse_loss(s);

	m->compile();

	pv.randomize();
	pv.normalize();

	s0->unroll(4);
	l->unroll(4, 4);
	s1->unroll(4);

	const int checkpoint_interval = 10000;

	for (int epoch = 0; true; epoch++) {
		m->cycle(x0, y0);
		if (epoch % checkpoint_interval == 0)
			std::cout << "S0: " << m->m_y.to_string() << std::endl;
		m->cycle(x1, y1);
		if (epoch % checkpoint_interval == 0)
			std::cout << "S1: " << m->m_y.to_string() << std::endl;
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
	double learn_rate = 0.1;

	for (int epoch = 0; true; epoch++) {
		double index_grad = get_index_grad(precision, index);
		index -= learn_rate * index_grad;
		precision += 0.000001;
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

	Mse_loss m = new mse_loss(p);

	m->compile();

	uniform_real_distribution<double> urd(-100, 100);
	p->m_x.pop(tensor::new_2d(2, units, urd, aurora::static_vals::random_engine));

	tensor desired = 1;

	for (int epoch = 0; true; epoch++) {

		m->cycle(p->m_x, desired);

		tensor update = p->m_x_grad.mul_2d(tensor::new_2d(2, units, 2));

		p->m_x.sub_2d(update, p->m_x);

		if (epoch % 1000 == 0) {
			std::cout << p->m_x.to_string() << std::endl << std::endl;
			std::cout << p->m_y.to_string() << std::endl;
		}

	}

}

void ntm_sparsify_test() {

	size_t memory_height = 5;

	ptr<ntm_sparsify> p = new ntm_sparsify(memory_height);

	Mse_loss m = new mse_loss(p);

	m->compile();


	uniform_real_distribution<double> urd(0, 1);

	p->m_x.pop(tensor::new_1d(memory_height, urd, aurora::static_vals::random_engine));
		
	double beta_des = 3;

	tensor x = tensor::new_1d(memory_height, urd, aurora::static_vals::random_engine);
	tensor y = tensor::new_1d(memory_height);
	for (int i = 0; i < x.size(); i++)
		y[i].val() = exp(beta_des * x[i]);

	for (int epoch = 0; true; epoch++) {

		m->cycle(x, y);

		p->m_beta[0].val() -= 0.0002 * p->m_beta_grad[0];

		// WE ARE SLEEPING HERE
		Sleep(10);

		if (epoch % 10 == 0) {
			std::cout << y.to_string() << std::endl << p->m_y.to_string() << std::endl;
			std::cout << p->m_beta.to_string() << std::endl << std::endl;
		}

	}

}

void normalize_test() {

	size_t units = 5;

	ptr<normalize> p = new normalize(units);

	Mse_loss m = new mse_loss(p);

	m->compile();

	uniform_real_distribution<double> urd(-10, 10);

	p->m_x.pop(tensor::new_1d(units, urd, aurora::static_vals::random_engine));
	tensor y = { 0.5, 0.1, 0.1, 0.1, 0.2 };

	tensor lr_tensor = tensor::new_1d(units, 0.02);

	for (int epoch = 0; true; epoch++) {

		m->cycle(p->m_x, y);

		tensor update = p->m_x_grad.mul_1d(lr_tensor);

		p->m_x.sub_1d(update, p->m_x);

		if (epoch % 10000 == 0) {
			std::cout << y.to_string() << std::endl << p->m_y.to_string() << std::endl << std::endl;
		}

	}
}

void ntm_content_addresser_test() {

	size_t memory_height = 5;
	size_t memory_width = 10;

	ptr<ntm_content_addresser> p = new ntm_content_addresser(memory_height, memory_width);

	Mse_loss m = new mse_loss(p);

	m->compile();

	uniform_real_distribution<double> urd(-1, 1);

	p->m_key.pop(tensor::new_1d(memory_width, urd, aurora::static_vals::random_engine));
	p->m_beta.pop(tensor::new_1d(1, urd, aurora::static_vals::random_engine));
	p->m_x.pop(tensor::new_2d(memory_height, memory_width, urd, aurora::static_vals::random_engine));

	tensor y = tensor::new_1d(memory_height);

	const size_t index_to_see = 1;
	y[index_to_see].val() = 1;

	const double lr = 0.02;
	tensor beta_lr_tensor = tensor::new_1d(1, lr);
	tensor key_lr_tensor = tensor::new_1d(memory_width, lr);
	//tensor x_lr_tensor = tensor::new_2d(memory_height, memory_width, lr);

	for (int epoch = 0; true; epoch++) {

		m->cycle(p->m_x, y);

		tensor beta_update = p->m_beta_grad.mul_1d(beta_lr_tensor);
		tensor key_update = p->m_key_grad.mul_1d(key_lr_tensor);
		//tensor x_update = p->x_grad.mul_2d(x_lr_tensor);

		p->m_beta.sub_1d(beta_update, p->m_beta);
		p->m_key.sub_1d(key_update, p->m_key);
		//p->x.sub_2d(x_update, p->x);

		if (epoch % 1000 == 0) {
			std::cout <<
				"INDEX: " << std::to_string(index_to_see) << std::endl <<
				p->m_x[index_to_see].to_string() << std::endl <<
				p->m_key.to_string() << std::endl <<
				p->m_y.to_string() << std::endl <<
				p->m_beta[0].to_string() << std::endl <<
				p->m_key.cos_sim(p->m_x[index_to_see]) << std::endl <<
				std::endl;
		}

	}

}

void interpolate_test() {

	size_t units = 5;

	ptr<layer> l_softmax = new layer(1, new sigmoid());
	l_softmax->compile();
	ptr<interpolate> l_interpolate = new interpolate(units);

	Mse_loss m = new mse_loss(l_interpolate);

	m->compile();

	l_softmax->m_y.link(l_interpolate->m_amount);
	l_softmax->m_y_grad.link(l_interpolate->m_amount_grad);

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
		m->cycle(x, y);
		l_softmax->bwd();

		tensor update = l_softmax->m_x_grad.mul_1d(lr_tensor);

		l_softmax->m_x.sub_1d(update, l_softmax->m_x);

		Sleep(10);

		if (epoch % 100 == 0)
			std::cout << l_interpolate->m_amount.to_string() << std::endl;

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

	Mse_loss m = new mse_loss(l_shift);

	m->compile();

	l_softmax->m_y.link(l_shift->m_amount);
	l_softmax->m_y_grad.link(l_shift->m_amount_grad);

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
		m->cycle(x, y);
		l_softmax->bwd();

		tensor update = l_softmax->m_x_grad.mul_1d(lr_tensor);
		l_softmax->m_x.sub_1d(update, l_softmax->m_x);

		if (epoch % 1000 == 0)
			std::cout << l_shift->m_amount.to_string() << std::endl;

	}

}

void power_test() {

	size_t units = 5;

	ptr<power> p = new power(units);

	Mse_loss m = new mse_loss(p);

	m->compile();

	uniform_real_distribution<double> urd(0, 1);

	const double amount_des = 3;

	tensor x = tensor::new_1d(units, urd, aurora::static_vals::random_engine);
	tensor y = tensor::new_1d(units);
	for (int i = 0; i < units; i++)
		y[i].val() = pow(x[i], amount_des);

	tensor lr_tensor = tensor::new_1d(1, 0.02);

	for (int epoch = 0; true; epoch++) {

		m->cycle(x, y);

		tensor update = p->m_amount_grad.mul_1d(lr_tensor);

		p->m_amount.sub_1d(update, p->m_amount);

		if (epoch % 10000 == 0)
			std::cout << p->m_amount.to_string() << std::endl;

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

	Mse_loss m = new mse_loss(p);

	m->compile();

	l_g_sm->m_y.link(p->m_g);
	l_g_sm->m_y_grad.link(p->m_g_grad);

	l_s_sm->m_y.link(p->m_s);
	l_s_sm->m_y_grad.link(p->m_s_grad);

	uniform_real_distribution<double> urd(0, 1);

	p->m_g.pop(tensor::new_1d(1, urd, aurora::static_vals::random_engine));
	p->m_s.pop(tensor::new_1d(valid_shifts.size(), urd, aurora::static_vals::random_engine));
	p->m_gamma.pop(tensor::new_1d(1, urd, aurora::static_vals::random_engine));

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
		m->cycle(x, y);
		l_s_sm->bwd();
		l_g_sm->bwd();

		tensor g_update = l_g_sm->m_x_grad.mul_1d(g_lr);
		tensor s_update = l_s_sm->m_x_grad.mul_1d(s_lr);
		tensor gamma_update = p->m_gamma_grad.mul_1d(gamma_lr);
		//tensor x_update = p->x_grad.mul_1d(x_lr);

		l_g_sm->m_x.sub_1d(g_update, l_g_sm->m_x);
		l_s_sm->m_x.sub_1d(s_update, l_s_sm->m_x);
		p->m_gamma.sub_1d(gamma_update, p->m_gamma);
		//p->x.sub_1d(x_update, p->x);

		if (epoch % 1000 == 0)
			std::cout << y.to_string() << std::endl << p->m_y.to_string() 
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

	Mse_loss m = new mse_loss(p);

	m->compile();

	uniform_real_distribution<double> urd(-1, 1);
	uniform_real_distribution<double> sm_urd(0, 1);
	uniform_real_distribution<double> pos_urd(1, 3);

	const size_t prev_selected_index = 3;

	p->m_wx.pop(tensor::new_1d(memory_height));
	p->m_wx[prev_selected_index].val() = 1;

	p->m_x.pop(tensor::new_2d(memory_height, memory_width, urd, aurora::static_vals::random_engine));
	p->m_key.pop(tensor::new_1d(memory_width, urd, aurora::static_vals::random_engine));
	p->m_beta.pop(tensor::new_1d(1, urd, aurora::static_vals::random_engine));
	p->m_gamma.pop(tensor::new_1d(1, pos_urd, aurora::static_vals::random_engine));
	l_g_sm->m_x.pop(tensor::new_1d(1, sm_urd, aurora::static_vals::random_engine));
	l_s_sm->m_x.pop(tensor::new_1d(valid_shifts.size(), sm_urd, aurora::static_vals::random_engine));

	l_g_sm->m_y.link(p->m_g);
	l_g_sm->m_y_grad.link(p->m_g_grad);
	l_s_sm->m_y.link(p->m_s);
	l_s_sm->m_y_grad.link(p->m_s_grad);

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
		m->cycle(p->m_x, y);
		l_s_sm->bwd();
		l_g_sm->bwd();

		// NEVER UPDATE WX LIKE THIS. WX SHOULD BE BETWEEN 0 AND 1
		//tensor wx_update = p->wx_grad.mul_1d(wx_lr);
		tensor x_update = p->m_x_grad.mul_2d(x_lr);
		tensor key_update = p->m_key_grad.mul_1d(key_lr);
		tensor beta_update = p->m_beta_grad.mul_1d(beta_lr);
		tensor gamma_update = p->m_gamma_grad.mul_1d(gamma_lr);
		tensor g_update = l_g_sm->m_x_grad.mul_1d(g_lr);
		tensor s_update = l_s_sm->m_x_grad.mul_1d(s_lr);
		
		//p->wx.sub_1d(wx_update, p->wx);
		p->m_x.sub_2d(x_update, p->m_x);
		p->m_key.sub_1d(key_update, p->m_key);
		p->m_beta.sub_1d(beta_update, p->m_beta);
		p->m_gamma.sub_1d(gamma_update, p->m_gamma);
		l_g_sm->m_x.sub_1d(g_update, l_g_sm->m_x);
		l_s_sm->m_x.sub_1d(s_update, l_s_sm->m_x);

		if (epoch % 1000 == 0)
			std::cout << y.to_string() << std::endl << p->m_y.to_string()
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

	param_vector pv;
	uniform_real_distribution<double> urd(-1, 1);

	ntm_rh nrh = ntm_rh(5, { 6, 7 }, 3);
	nrh.param_recur(pseudo::param_init(new param_sgd(0.002), pv));
	nrh.compile();

	pv.randomize();
	pv.normalize();

	for (int epoch = 0; epoch < 1000000; epoch++) {

		double cost = 0;

			nrh.fwd(x_0);

			nrh.m_key_grad.pop(nrh.m_key.sub_1d(des_k_0));
			nrh.m_beta_grad.pop(nrh.m_beta.sub_1d(des_beta_0));
			nrh.m_gamma_grad.pop(nrh.m_gamma.sub_1d(des_gamma_0));
			nrh.m_g_grad.pop(nrh.m_g.sub_1d(des_g_0));
			nrh.m_s_grad.pop(nrh.m_s.sub_1d(des_s_0));

			nrh.bwd();

			cost +=
				nrh.m_key_grad.abs_1d().sum_1d() +
				nrh.m_beta_grad.abs_1d().sum_1d() +
				nrh.m_gamma_grad.abs_1d().sum_1d() +
				nrh.m_g_grad.abs_1d().sum_1d() +
				nrh.m_s_grad.abs_1d().sum_1d();

			nrh.fwd(x_1);

			nrh.m_key_grad.pop(nrh.m_key.sub_1d(des_k_1));
			nrh.m_beta_grad.pop(nrh.m_beta.sub_1d(des_beta_1));
			nrh.m_gamma_grad.pop(nrh.m_gamma.sub_1d(des_gamma_1));
			nrh.m_g_grad.pop(nrh.m_g.sub_1d(des_g_1));
			nrh.m_s_grad.pop(nrh.m_s.sub_1d(des_s_1));

			nrh.bwd();

			cost +=
				nrh.m_key_grad.abs_1d().sum_1d() +
				nrh.m_beta_grad.abs_1d().sum_1d() +
				nrh.m_gamma_grad.abs_1d().sum_1d() +
				nrh.m_g_grad.abs_1d().sum_1d() +
				nrh.m_s_grad.abs_1d().sum_1d();


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

	param_vector pv;
	uniform_real_distribution<double> urd(-1, 1);

	ntm_wh nwh = ntm_wh(5, { 6, 7 }, 3);
	nwh.param_recur(pseudo::param_init(new param_sgd(0.0002), pv));
	nwh.compile();
	
	pv.randomize();
	pv.normalize();

	for (int epoch = 0; epoch < 1000000; epoch++) {

		double cost = 0;

		nwh.fwd(x_0);

		nwh.m_internal_rh->m_key_grad.pop(nwh.m_internal_rh->m_key.sub_1d(des_k_0));
		nwh.m_internal_rh->m_beta_grad.pop(nwh.m_internal_rh->m_beta.sub_1d(des_beta_0));
		nwh.m_internal_rh->m_gamma_grad.pop(nwh.m_internal_rh->m_gamma.sub_1d(des_gamma_0));
		nwh.m_internal_rh->m_g_grad.pop(nwh.m_internal_rh->m_g.sub_1d(des_g_0));
		nwh.m_internal_rh->m_s_grad.pop(nwh.m_internal_rh->m_s.sub_1d(des_s_0));
		nwh.m_e_grad.pop(nwh.m_e.sub_1d(des_e_0));
		nwh.m_a_grad.pop(nwh.m_a.sub_1d(des_a_0));

		nwh.bwd();

		cost +=
			nwh.m_internal_rh->m_key_grad.abs_1d().sum_1d() +
			nwh.m_internal_rh->m_beta_grad.abs_1d().sum_1d() +
			nwh.m_internal_rh->m_gamma_grad.abs_1d().sum_1d() +
			nwh.m_internal_rh->m_g_grad.abs_1d().sum_1d() +
			nwh.m_internal_rh->m_s_grad.abs_1d().sum_1d() +
			nwh.m_e_grad.abs_1d().sum_1d() +
			nwh.m_a_grad.abs_1d().sum_1d();

		nwh.fwd(x_1);

		nwh.m_internal_rh->m_key_grad.pop(nwh.m_internal_rh->m_key.sub_1d(des_k_1));
		nwh.m_internal_rh->m_beta_grad.pop(nwh.m_internal_rh->m_beta.sub_1d(des_beta_1));
		nwh.m_internal_rh->m_gamma_grad.pop(nwh.m_internal_rh->m_gamma.sub_1d(des_gamma_1));
		nwh.m_internal_rh->m_g_grad.pop(nwh.m_internal_rh->m_g.sub_1d(des_g_1));
		nwh.m_internal_rh->m_s_grad.pop(nwh.m_internal_rh->m_s.sub_1d(des_s_1));
		nwh.m_e_grad.pop(nwh.m_e.sub_1d(des_e_0));
		nwh.m_a_grad.pop(nwh.m_a.sub_1d(des_a_0));

		nwh.bwd();

		cost +=
			nwh.m_internal_rh->m_key_grad.abs_1d().sum_1d() +
			nwh.m_internal_rh->m_beta_grad.abs_1d().sum_1d() +
			nwh.m_internal_rh->m_gamma_grad.abs_1d().sum_1d() +
			nwh.m_internal_rh->m_g_grad.abs_1d().sum_1d() +
			nwh.m_internal_rh->m_s_grad.abs_1d().sum_1d() +
			nwh.m_e_grad.abs_1d().sum_1d() +
			nwh.m_a_grad.abs_1d().sum_1d();;

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

	param_vector pv;
	auto pmt_init = pseudo::param_init(
		new param_sgd(0.002), pv);

	Ntm_reader p = new ntm_reader(memory_height, memory_width, valid_shifts, {memory_width, memory_width + 5});
	p->param_recur(pmt_init);

	Mse_loss m = new mse_loss(p);

	m->compile();

	pv.randomize();
	pv.normalize();

	p->m_mx.pop(tensor::new_2d(memory_height, memory_width, mem_urd, aurora::static_vals::random_engine));
	//p->wx[1].val() = 1;

	const size_t selected_index = 1;
	tensor y = p->m_mx[selected_index];

	/*tensor y = tensor::new_1d(memory_width);
	for (int i = 0; i < memory_width; i++)
		y[i].val() = 0.5*p->mx[2][i] + 0.5*p->mx[4][i];*/

	for (int epoch = 0; true; epoch++) {

		m->cycle(p->m_x, y);

		for (param_sgd* pmt : pv)
			pmt->update();

		if (epoch % 1000 == 0)
			std::cout <<
			"WY: " << p->m_internal_addresser->m_wy.to_string() << std::endl <<
			"DESIRED: " << y.to_string() << std::endl <<
			"ACTUAL:  " << p->m_y.to_string() << std::endl <<
			"GAMMA: " << p->m_internal_addresser->m_gamma[0].to_string() << std::endl << std::endl;

	}

}

// BLOWS UP, EXPECTED
void ntm_writer_test() {

	size_t memory_height = 5;
	size_t memory_width = 5;
	vector<int> valid_shifts = { -1, 0, 1 };

	uniform_real_distribution<double> pmt_urd(-0.1, 0.1);
	uniform_real_distribution<double> mem_urd(-10, 10);

	default_random_engine dre(27);

	param_vector pv;
	auto pmt_init = pseudo::param_init(
		new param_mom(0.00002, 0.9), pv);

	ptr<ntm_writer> p = new ntm_writer(memory_height, memory_width, valid_shifts, { memory_width });
	p->param_recur(pmt_init);

	Mse_loss m = new mse_loss(p);

	m->compile();

	pv.randomize();
	pv.normalize();

	p->m_mx.pop(tensor::new_2d(memory_height, memory_width, mem_urd, aurora::static_vals::random_engine));
	p->m_wx[4].val() = 0.4;
	p->m_wx[3].val() = 0.6;

	const size_t selected_index = 1;
	tensor y = p->m_mx.clone();
	y[selected_index].add_1d(tensor::new_1d(memory_width, 1), y[selected_index]);

	/*tensor y = tensor::new_1d(memory_width);
	for (int i = 0; i < memory_width; i++)
		y[i].val() = 0.5*p->mx[2][i] + 0.5*p->mx[4][i];*/

	for (int epoch = 0; true; epoch++) {

		m->cycle(p->m_x, y);

		for (param_mom* pmt : pv)
			pmt->update();

		double cost = p->m_y_grad.abs_2d().sum_2d().sum_1d();

		if (epoch % 10000 == 0)
			std::cout << std::to_string(cost) << std::endl <<
			"G: " << p->m_internal_addresser->m_g[0].to_string() << std::endl <<
			"S: " << p->m_internal_addresser->m_s.to_string() << std::endl <<
			"GAMMA: " << p->m_internal_addresser->m_gamma[0].to_string() << std::endl <<
			"BETA:     " << p->m_internal_addresser->m_beta[0].to_string() << std::endl <<
			"WY: " << p->m_wy.to_string() << std::endl <<
			"SIM_TENSOR: " << p->m_internal_addresser->m_internal_content_addresser->m_internal_similarity->m_y.to_string() << std::endl <<
			"A: " << p->m_internal_head->m_a.to_string() << std::endl <<
			"DES Y: " << y[selected_index].to_string() << std::endl <<
			"ACT Y: " << p->m_y[selected_index].to_string() << std::endl <<
			std::endl << std::endl;
	}

}

void ntm_test() {

	size_t memory_height = 1;
	size_t memory_width = 5;
	size_t num_readers = 1;
	size_t num_writers = 1;
	vector<int> valid_shifts = { -1, 0, 1 };
	vector<size_t> head_hidden_dims = {  };

	uniform_real_distribution<double> pmt_urd(-1, 1);
	uniform_real_distribution<double> ts_urd(-10, 10);
	uniform_real_distribution<double> mem_urd(-1, 1);
	default_random_engine dre(27);

	param_vector pv;

	auto pmt_init = pseudo::param_init(new param_mom(1, 0.9), pv);

	Sync s_in = new sync(pseudo::tnn({ 2, memory_width }, pseudo::nlr(0.3)));

	Ntm p = new ntm(
		memory_height,
		memory_width,
		num_readers,
		num_writers,
		valid_shifts,
		head_hidden_dims);

	Sync s_out = new sync(pseudo::tnn({ memory_width, 1 }, pseudo::nsm()));

	Sequential s = new sequential({ s_in, p, s_out });
	s->param_recur(pmt_init);

	const size_t num_ts = 4;

	s_in->prep(num_ts);
	p->prep(num_ts);
	s_out->prep(num_ts);

	Mse_loss m = new mse_loss(s);

	m->compile();

	pv.randomize();
	pv.normalize();

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

	const size_t checkpoint_interval = 100;

	double cost = INFINITY;

	for (int epoch = 0; cost > 0.1; epoch++) {

		cost = 0;

		for (int ts = 0; ts < x.size(); ts++) {
			m->cycle(x[ts], y[ts]);
			cost += s->m_y_grad.abs_2d().sum_2d().sum_1d();

			if (epoch % checkpoint_interval == 0)
				std::cout << s->m_y.to_string() << std::endl;
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

	param_vector pv;

	auto pmt_init = [&](Param& pmt) {
		pmt = new param_sgd(0.02);
		pv.push_back(pmt);
	};
	Stacked_recurrent s = new stacked_recurrent({
		new sync(pseudo::tnn({ 2, lstm_units }, pseudo::nlr(0.3))),
		new lstm(lstm_units),
		new sync(pseudo::tnn({ lstm_units, 1 }, pseudo::nlr(0.3)))
	});
	s->param_recur(pmt_init);

	s->prep(4);

	Mse_loss m = new mse_loss(s);

	m->compile();

	pv.randomize();
	pv.normalize();

	s->unroll(4);

	const int checkpoint_interval = 10000;

	for (int epoch = 0; true; epoch++) {
		m->cycle(x0, y0);
		if (epoch % checkpoint_interval == 0)
			std::cout << "S0: " << s->m_y.to_string() << std::endl;
		m->cycle(x1, y1);
		if (epoch % checkpoint_interval == 0)
			std::cout << "S1: " << s->m_y.to_string() << std::endl;
		for (param_sgd* pmt : pv) {
			pmt->state() -= pmt->learn_rate() * pmt->gradient();
			pmt->gradient() = 0;
		}
	}

}

void lstm_mdim_test() {

	param_vector pv;
	Stacked_recurrent s = pseudo::lstm_mdim(2, 5, 1);

	Mse_loss m = new mse_loss(s);

	m->param_recur(pseudo::param_init(new param_mom(0.02, 0.9), pv));
	pv.rand_norm();

	s->prep(4);
	m->compile();
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

	for (int epoch = 0; epoch < 100000; epoch++)
	{
		m->cycle(x0, y0);
		if (epoch % checkpoint_interval == 0)
			std::cout << "S0: " << s->m_y.to_string() << std::endl;
		m->cycle(x1, y1);
		if (epoch % checkpoint_interval == 0)
			std::cout << "S1: " << s->m_y.to_string() << std::endl;
		pv.update();
	}

}

void lstm_stacked_mdim_test() {

	param_vector pv;
	Stacked_recurrent s = pseudo::lstm_stacked_mdim(2, 10, 1, 1);

	s->prep(4);

	s->param_recur(pseudo::param_init(new param_mom(0.02, 0.9), pv));
	pv.rand_norm();

	Mse_loss m = new mse_loss(s);
	m->compile();

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
		m->cycle(x0, y0);
		if (epoch % checkpoint_interval == 0)
			std::cout << "S0: " << s->m_y.to_string() << std::endl;
		m->cycle(x1, y1);
		if (epoch % checkpoint_interval == 0)
			std::cout << "S1: " << s->m_y.to_string() << std::endl;
		pv.update();
	}



}

void test_test() {

	param* p = new param();
	Param p1 = p;
	Param p2 = new param();

	test_test();


}

void ntm_mdim_test() {

	param_vector param_vec;
	Stacked_recurrent s = pseudo::ntm_mdim(2, 1, 10, 5, 1, 1, { -1, 0, 1 }, { 5, 10 });

	s->prep(4);

	Mse_loss m = new mse_loss(s);
	m->compile();

	m->param_recur(pseudo::param_init(new param_mom(0.02, 0.9), param_vec));
	param_vec.rand_norm();

	s->unroll(4);

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

		const size_t MINI_BATCH_SIZE = 4;

		for (int mb_ts = 0; mb_ts < MINI_BATCH_SIZE; mb_ts++) {
			size_t ts = random(0, x.size());
			m->cycle(x[ts], y[ts]);
			cost += s->m_y_grad.abs_2d().sum_2d().sum_1d();
		}

		param_vec.update();

		if (epoch % checkpoint_interval == 0)
			std::cout << cost << std::endl;
	}

}

void ptr_test() {

	ptr<int> p1;
	{
		ptr<int> p2 = new int(10);
		p1 = p2;
	}
	ptr<int> p3 = p1;

}

void tensor_noncompiled_test() {
	tensor t1 = tensor::new_2d(10, 10);
	tensor t2;
	for (int i = 0; true; i++) {
		if (i % 1000000 == 0)
			printf("");
		t2 = t1;
	}
}

void major_tests() {
	tnn_xor_test();
	lstm_test();
	ntm_test();
	lstm_mdim_test();
	cnl_test();
}

std::uniform_real_distribution<> uniform_zero_to_one(0.0, 1.0);
bool random_bool_with_prob(const double& prob)  // probability between 0.0 and 1.0
{
	return uniform_zero_to_one(aurora::static_vals::random_engine) >= prob;
}





























tensor get_random_integer_tensor_1d(int a_min, int a_max, size_t a_size) {
	uniform_int_distribution<int> uid(a_min, a_max);
	tensor result = tensor::new_1d(a_size);
	for (int i = 0; i < a_size; i++) {
		result[i] = uid(aurora::static_vals::random_engine);
	}
	return result;
}

tensor get_random_integer_tensor_2d(int a_min, int a_max, size_t a_height, size_t a_width) {
	tensor result = tensor::new_1d(a_height);
	for (int i = 0; i < a_height; i++) {
		result[i] = get_random_integer_tensor_1d(a_min, a_max, a_width);
	}
	return result;
}

tensor get_rounded_1d(tensor a_x) {
	tensor result = tensor::new_1d(a_x.size());
	for (int i = 0; i < a_x.size(); i++) {
		result[i] = round(a_x[i]);
	}
	return result;
}

void shuffle_tensor(tensor& a_x) {
	tensor temp = tensor::new_1d(a_x.size());
	vector<int> valid_dst(a_x.size());
	for (int i = 0; i < a_x.size(); i++)
		valid_dst[i] = i;
	for (int i = 0; i < a_x.size(); i++) {
		int selected_dst_index = rand() % valid_dst.size();
		temp[valid_dst[selected_dst_index]] = a_x[i];
		valid_dst.erase(valid_dst.begin() + selected_dst_index);
	}
	a_x.pop(temp);
}

template<typename CONTAINER_TYPE>
void append_params_to_file(const string& a_file_name, CONTAINER_TYPE a_param_container) {
	ofstream ofs(a_file_name, std::ios::app | std::ios::out);
	ofs.precision(16);
	for (int i = 0; i < a_param_container.size(); i++) {
		ofs << a_param_container[i]->state() << ",";
	}
	ofs << "\n";
	ofs.close();
}

void tnn_xor_test_param_export() {

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

	param_vector pl;

	uniform_real_distribution<double> urd(-1, 1);
	default_random_engine re(25);

	Sequential s = pseudo::tnn({ 2, 5, 1 }, pseudo::nlr(0.3));
	s->param_recur([&](Param& pmt) {
		pmt = new param_mom(0.02, 0.9);
		pl.push_back((param_mom*)pmt.get());
	});

	Mse_loss m = new mse_loss(s);

	m->compile();

	pl.randomize();
	pl.normalize();

	printf("");

	const size_t checkpoint_interval = 10;
	
	// NORMALIZE PARAMETERS
	tensor l_param_tensor = tensor::new_1d(pl.size());
	
	for (int i = 0; i < pl.size(); i++)
		l_param_tensor[i].val() = pl[i]->state();

	l_param_tensor.norm_1d(l_param_tensor);

	for (int i = 0; i < l_param_tensor.size(); i++)
		pl[i]->state() = l_param_tensor[i].val();

	for (int epoch = 0; epoch < 1000000; epoch++) {

		if (epoch % checkpoint_interval == 0) {
			printf("\033[%d;%dH", 0, 0);
			std::cout << epoch << std::endl;
			append_params_to_file("tnn_xor_params_data.txt", pl);
		}

		for (int tsIndex = 0; tsIndex < x.size(); tsIndex++) {
			m->cycle(x[tsIndex], y[tsIndex]);
			if (epoch % checkpoint_interval == 0)
				std::cout << x[tsIndex].to_string() << " " << s->m_y.to_string() << std::endl;
		}

		for (param_mom* pmt : pl) {
			pmt->update();
		}
	}

	for (param_mom* pmt : pl) {
		std::cout << pmt->state() << std::endl;
	}

}

int collapse_sp_onehot_index(const tensor& a_probability_tensor) {
	tensor result = tensor::new_1d(a_probability_tensor.size());
	double random_value = random_d(0, 1, static_vals::random_engine);
	double bin_lower_bound = 0;
	double bin_upper_bound = 0;
	for (int i = 0; i < a_probability_tensor.size(); i++) {
		bin_upper_bound += a_probability_tensor[i];
		if (random_value >= bin_lower_bound && random_value <= bin_upper_bound) {
			return i;
		}
		bin_lower_bound += a_probability_tensor[i];
	}
	return -1;
}

int collapse_zero_dimensional_sp(const double& a_probability_of_zero) {
	if (random_bool_with_prob(a_probability_of_zero)) {
		return 0;
	}
	else {
		return 1;
	}
}

tensor collapse_sp(const tensor& a_probability_tensor) {
	int selected_index = collapse_sp_onehot_index(a_probability_tensor);
	tensor result = tensor::new_1d(a_probability_tensor.size());
	result[selected_index].val() = 1;
	return result;
}

double sign_d(double a_x) {
	if (a_x >= 0) return 1;
	else return -1;
}

double sech_d(double a_x) {
	return 1.0 / cosh(a_x);
}

double sigmoid_d(double a_x) {
	return 1.0 / (1.0 + exp(-a_x));
}

void spc_raw_test() {
	
	const int SP_SIZE = 4;

	tensor x = tensor::new_1d(SP_SIZE);
	tensor x_error = tensor::new_1d(SP_SIZE);

	Sequential s = new sequential({ new layer(SP_SIZE, new sigmoid()), new normalize(SP_SIZE) });
	s->compile();

	x.link(s->m_x);
	x_error.link(s->m_x_grad);

	tensor p = tensor::new_1d(x.size());
	p.link(s->m_y);

	tensor p_error = tensor::new_1d(x.size());
	p_error.link(s->m_y_grad);

	// starting x values (not probabilities, but will be converted to (+) values through sigmoid, and then normalized to make probabilities.)
	uniform_real_distribution<double> x_urd(-3, 3);
	x.pop(tensor::new_1d(SP_SIZE, x_urd, static_vals::random_engine));

	uniform_real_distribution<double> v_urd(-10, 10);

	tensor desired_output = tensor::new_1d(SP_SIZE);
	desired_output[0].val() = 1;
	
	tensor predicted_output = tensor::new_1d(SP_SIZE);

	auto fwd = [&] {
		s->fwd();
		predicted_output = collapse_sp(p);
	};

	auto get_error = [&]() {
		return predicted_output.sub_1d(desired_output);
	};

	auto bwd = [&](tensor a_error) {
		for (int i = 0; i < p_error.size(); i++) {
			p_error[i].val() = a_error[i];
		}
		s->bwd();
	};
	
	// TRAINING OBJECTS
	tensor learn_rate_tensor = tensor::new_1d(x.size(), 0.2);

	for (int i = 0; true; i++) {

		fwd();
		tensor error = get_error();
		bwd(error);

		tensor update_tensor = learn_rate_tensor.mul_1d(x_error);
		x.sub_1d(update_tensor, x);

		if (i % 1000 == 0) {
			std::cout << p.to_string() << std::endl;
		}

	}

}

void binary_choice_test() {

	const int SP_SIZE = 5;

	tensor x = 0;
	tensor x_error = 0;

	Sigmoid s = new sigmoid();
	s->compile();

	x.link(s->m_x);
	x_error.link(s->m_x_grad);

	tensor p = 0;
	p.link(s->m_y);

	tensor p_error = 0;
	p_error.link(s->m_y_grad);

	x.pop(0.5);

	double desired_output = 0;
	double selected_output = 0;

	auto predict = [&] {
		s->fwd();
		selected_output = collapse_zero_dimensional_sp(p);
		return (double)selected_output;
	};

	auto get_error = [&](double a_prediction) {
		return a_prediction - desired_output;
	};

	auto back_propagate = [&](double a_error) {
		p_error.val() = a_error;
		s->bwd();
	};


	for (int i = 0; true; i++) {

		double prediction = predict();
		double error = get_error(prediction);
		back_propagate(error);

		x.val() -= 0.02 * x_error.val();

		if (i % 1000 == 0) {
			std::cout << p.to_string() << std::endl;
		}

	}

}

void spc_model_test() {

	const int SP_SIZE = 4;

	Sequential s = new sequential({new layer(SP_SIZE, new sigmoid()), new normalize(SP_SIZE), new onehot_spc(SP_SIZE)});

	Mse_loss m = new mse_loss(s);

	m->compile();

	tensor desired_output = tensor::new_1d(SP_SIZE);
	desired_output[3].val() = 1;

	tensor learn_rate_tensor = tensor::new_1d(SP_SIZE, 0.2);

	for (int i = 0; true; i++) {

		m->cycle(s->m_x, desired_output);

		tensor update_tensor = learn_rate_tensor.mul_1d(s->m_x_grad);
		s->m_x.sub_1d(update_tensor, s->m_x);

		if (i % 1000 == 0) {
			std::cout << ((onehot_spc*)s->m_models.back())->m_x.to_string() << std::endl;
		}

	}

}

void tensor_linkage_compute_time_test()
{
	Sequential seq = pseudo::tnn({ 100, 1000, 100 }, pseudo::nlr(0.3));
	std::cout << "Instantiated." << std::endl;
	seq->compile();
	std::cout << "Compiled." << std::endl;
}

void test_function_call_speed(const std::function<void(int)>& a_func, size_t a_recur = 1)
{
	if (a_recur == 0)
		return;
	a_func(10);
	test_function_call_speed(a_func, a_recur - 1);
}

void test_rounding_spc()
{
	Rounding_spc r = new rounding_spc();

	Mse_loss m = new mse_loss(r);

	m->compile();

	tensor x = 100;
	tensor y = 2;

	for (int epoch = 0; true; epoch++)
	{
		m->cycle(x, y);
		x.val() -= 0.002 * m->m_x_grad.val();
		if (epoch % 1000 == 0)
			std::cout << m->m_y.val() << std::endl;
	}

}

void example_tnn_setup()
{
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

	param_vector pv;
	Sequential s = pseudo::tnn({ 2, 5, 1 }, pseudo::nlr(0.3));

	Mse_loss m = new mse_loss(s);
	m->compile();

	m->param_recur(pseudo::param_init(new param_mom(0.02, 0.9), pv));
	pv.rand_norm();
	
	for (int epoch = 0; epoch < 1000; epoch++)
	{
		double cost = 0;
		for (int ts = 0; ts < x.size(); ts++)
		{
			m->cycle(x[ts], y[ts]);
			cost += s->m_y_grad.abs_1d().sum_1d();
		}

		pv.update();

		if (epoch % 999 == 0)
			std::cout << cost << std::endl;

	}

}

void test_large_model_linkage()
{

	Sequential s = pseudo::tnn({ 100, 1000, 100 }, pseudo::nlr(0.3));

	affix_base::timing::stopwatch l_stopwatch;
	l_stopwatch.start();

	s->compile();
	std::cout << l_stopwatch.duration_milliseconds();

}

int main() {

	srand(time(NULL));

	major_tests();

	return 0;

}
