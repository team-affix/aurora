#include "ready.h"

using namespace aurora;

ptr<model> basic::ready_tnn(vector<size_t> a_npl, ptr<model> a_neuron_template, vector<param*> a_pv) {
	ptr<model> result = pseudo::tnn(a_npl, a_neuron_template, pseudo::dump_pmt(a_pv));
	uniform_real_distribution<double> state = pseudo::state(a_pv.size());
	result->pmt_wise(pseudo::init_pmt(state));
	result->compile();
	return result;
}

ptr<model> basic::ready_tnn(vector<size_t> a_npl, vector<param*> a_pv) {
	return ready_tnn(a_npl, pseudo::nlr(0.3), a_pv);
}

ptr<model> basic::ready_tnn(tensor& a_x, tensor& a_y, vector<param*> a_pv) {
	return ready_tnn({ a_x.width(), a_x.width() + a_y.width(), a_y.width() }, a_pv);
}

