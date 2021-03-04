#pragma once
#include "param.h"
#include "param_sgd.h"
#include "param_mom.h"
#include "param_sgd_mt.h"
#include "param_mom_mt.h"
#include <random>

using namespace aurora::params;
using std::uniform_real_distribution;

namespace aurora {
	namespace pseudo {
		function<void(ptr<param>&)> init_pmt(uniform_real_distribution<double> a_urd);
		function<void(ptr<param>&)> init_pmt(uniform_real_distribution<double> a_urd, double a_learn_rate);
		function<void(ptr<param>&)> init_pmt(uniform_real_distribution<double> a_urd, double a_learn_rate, double a_beta);
		function<void(ptr<param>&)> init_pmt(uniform_real_distribution<double> a_urd, vector<param*>& a_pv);
		function<void(ptr<param>&)> init_pmt(uniform_real_distribution<double> a_urd, double a_learn_rate, vector<param_sgd*>& a_pv);
		function<void(ptr<param>&)> init_pmt(uniform_real_distribution<double> a_urd, double a_learn_rate, double a_beta, vector<param_mom*>& a_pv);
		function<void(ptr<param>&)> dump_pmt(vector<param*>& a_pv);
		function<void(ptr<param>&)> dump_pmt(vector<param_sgd*>& a_pv);
		function<void(ptr<param>&)> dump_pmt(vector<param_mom*>& a_pv);
	}
}