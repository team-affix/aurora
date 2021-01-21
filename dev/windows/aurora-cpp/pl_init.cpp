#include "pl_init.h"
#include "random.h"

using aurora::math::random;
using namespace aurora;

void pseudo::pl_init(vector<param*>& a_pl, int a_rnd_init, double a_state_min, double a_state_max, double a_learn_rate) {
	random rnd(a_rnd_init);
	for (param* pmt : a_pl) {
		pmt->state() = rnd.next_double(a_state_min, a_state_max);
		pmt->learn_rate() = a_learn_rate;
	}
}
void pseudo::pl_init(vector<param_sgd*>& a_pl, int a_rnd_init, double a_state_min, double a_state_max, double a_learn_rate) {
	random rnd(a_rnd_init);
	for (param_sgd* pmt : a_pl) {
		pmt->state() = rnd.next_double(a_state_min, a_state_max);
		pmt->learn_rate() = a_learn_rate;
	}
}
void pseudo::pl_init(vector<param_mom*>& a_pl, int a_rnd_init, double a_state_min, double a_state_max, double a_learn_rate, double a_beta) {
	random rnd(a_rnd_init);
	for (param_mom* pmt : a_pl) {
		pmt->state() = rnd.next_double(a_state_min, a_state_max);
		pmt->learn_rate() = a_learn_rate;
		pmt->beta() = a_beta;
	}
}