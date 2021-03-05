#pragma once
#include "pseudo.h"

using namespace aurora;

namespace aurora {
	namespace basic {
		ptr<model> ready_tnn(vector<size_t> a_npl, ptr<model> a_neuron_template, vector<param*> a_pv);
		ptr<model> ready_tnn(vector<size_t> a_npl, ptr<model> a_neuron_template, vector<param_sgd*> a_pv);
		ptr<model> ready_tnn(vector<size_t> a_npl, ptr<model> a_neuron_template, vector<param_mom*> a_pv);
		ptr<model> ready_tnn(vector<size_t> a_npl, ptr<model> a_neuron_template, vector<param_sgd_mt*> a_pv);
		ptr<model> ready_tnn(vector<size_t> a_npl, ptr<model> a_neuron_template, vector<param_mom_mt*> a_pv);
		ptr<model> ready_tnn(vector<size_t> a_npl, vector<param*> a_pv);
		ptr<model> ready_tnn(vector<size_t> a_npl, vector<param_sgd*> a_pv);
		ptr<model> ready_tnn(vector<size_t> a_npl, vector<param_mom*> a_pv);
		ptr<model> ready_tnn(vector<size_t> a_npl, vector<param_sgd_mt*> a_pv);
		ptr<model> ready_tnn(vector<size_t> a_npl, vector<param_mom_mt*> a_pv);
		ptr<model> ready_tnn(tensor& a_x, tensor& a_y, vector<param*> a_pv);
	}
}