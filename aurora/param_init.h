#pragma once
#include "pch.h"
#include "param.h"
#include "param_sgd.h"
#include "param_mom.h"
#include "param_sgd_mt.h"
#include "param_mom_mt.h"

using namespace aurora::params;
using std::uniform_real_distribution;

namespace aurora {
	namespace initialization {
		/*double state_structure(size_t a_pv_len);
		uniform_real_distribution<double> state(size_t a_pv_len);
		double learn_rate(size_t a_x_len);*/
		function<void(ptr<param>&)> init_pmt(uniform_real_distribution<double> a_urd);
		function<void(ptr<param>&)> init_pmt(uniform_real_distribution<double> a_urd, double a_learn_rate);
		function<void(ptr<param>&)> init_pmt(uniform_real_distribution<double> a_urd, double a_learn_rate, double a_beta);
		function<void(ptr<param>&)> init_pmt(uniform_real_distribution<double> a_urd, vector<param*>& a_pv);
		function<void(ptr<param>&)> init_pmt(uniform_real_distribution<double> a_urd, double a_learn_rate, vector<param_sgd*>& a_pv);
		function<void(ptr<param>&)> init_pmt(uniform_real_distribution<double> a_urd, double a_learn_rate, double a_beta, vector<param_mom*>& a_pv);
		template<class T>
		function<void(ptr<param>&)> dump_pmt(vector<T*>& a_pv) {
			return [&](ptr<param>& pmt) {
				pmt = new T();
				a_pv.push_back((T*)pmt.get());
			};
		}
	}
}