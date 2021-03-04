#include "param_init.h"
#include "static.h"

using namespace aurora;

function<void(ptr<param>&)> pseudo::init_pmt(uniform_real_distribution<double> a_urd) {
	return [&](ptr<param>& pmt) {
		pmt = new param(a_urd(static_vals::aurora_random_engine));
	};
}

function<void(ptr<param>&)> pseudo::init_pmt(uniform_real_distribution<double> a_urd, double a_learn_rate) {
	return [&](ptr<param>& pmt) {
		pmt = new param_sgd(a_urd(static_vals::aurora_random_engine), a_learn_rate, 0);
	};
}

function<void(ptr<param>&)> pseudo::init_pmt(uniform_real_distribution<double> a_urd, double a_learn_rate, double a_beta) {
	return [&](ptr<param>& pmt) {
		pmt = new param_mom(a_urd(static_vals::aurora_random_engine), a_learn_rate, 0, 0, a_beta);
	};
}

function<void(ptr<param>&)> pseudo::init_pmt(uniform_real_distribution<double> a_urd, vector<param*>& a_pv) {
	return [&](ptr<param>& pmt) {
		pmt = new param(a_urd(static_vals::aurora_random_engine));
		a_pv.push_back(pmt.get());
	};
}

function<void(ptr<param>&)> pseudo::init_pmt(uniform_real_distribution<double> a_urd, double a_learn_rate, vector<param_sgd*>& a_pv) {
	return [&](ptr<param>& pmt) {
		pmt = new param_sgd(a_urd(static_vals::aurora_random_engine), a_learn_rate, 0);
		a_pv.push_back((param_sgd*)pmt.get());
	};
}

function<void(ptr<param>&)> pseudo::init_pmt(uniform_real_distribution<double> a_urd, double a_learn_rate, double a_beta, vector<param_mom*>& a_pv) {
	return [&](ptr<param>& pmt) {
		pmt = new param_mom(a_urd(static_vals::aurora_random_engine), a_learn_rate, 0, 0, a_beta);
		a_pv.push_back((param_mom*)pmt.get());
	};
}

function<void(ptr<param>&)> pseudo::dump_pmt(vector<param*>& a_pv) {
	return [&](ptr<param>& pmt) {
		pmt = new param();
		a_pv.push_back(pmt.get());
	};
}

function<void(ptr<param>&)> pseudo::dump_pmt(vector<param_sgd*>& a_pv) {
	return [&](ptr<param>& pmt) {
		pmt = new param_sgd();
		a_pv.push_back((param_sgd*)pmt.get());
	};
}

function<void(ptr<param>&)> pseudo::dump_pmt(vector<param_mom*>& a_pv) {
	return [&](ptr<param>& pmt) {
		pmt = new param_mom();
		a_pv.push_back((param_mom*)pmt.get());
	};
}
