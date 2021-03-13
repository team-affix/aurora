#include "rnc.h"

using aurora::models::rnc;

rnc::rnc() {

}

rnc::rnc(size_t a_units, function<void(ptr<param>&)> a_init) {
	this->units = a_units;
	this->weak_interface = new lstm_ts(units, a_init);
	this->strong_interface = new lstm_ts(units, a_init);
	this->s = tensor::new_2d(0, a_units);
}

rnc::rnc(size_t a_units, ptr<lstm_ts> a_weak_interface, ptr<lstm_ts> a_strong_interface, tensor a_s) {
	this->units = a_units;
	this->weak_interface = a_weak_interface;
	this->strong_interface = a_strong_interface;
	this->s = a_s;
}

void rnc::pmt_wise(function<void(ptr<param>&)> a_func) {
	weak_interface->pmt_wise(a_func);
	strong_interface->pmt_wise(a_func);
}

model* rnc::clone() {
	rnc* result = new rnc(units, (lstm_ts*)weak_interface->clone(), (lstm_ts*)strong_interface->clone(), s.clone());
	return result;
}

model* rnc::clone(function<void(ptr<param>&)> a_init) {
	rnc* result = new rnc(units, (lstm_ts*)weak_interface->clone(a_init), (lstm_ts*)strong_interface->clone(a_init), s.clone());
	return result;
}

void rnc::fwd() {
	// TREAT X AS TENSOR 2D
	weak_interface->fwd();
	tensor sim_tensor = tensor::new_1d(s.height());
	for (int slot = 0; slot < s.height(); slot++)
		sim_tensor[slot] = weak_interface->y.cos_sim(s[slot]);

	const double MAX_SIM_FOR_ALLOC = 0.4;

	int slot_index = sim_tensor.arg_max();
	if (slot_index == -1) {
		s.vec().push_back(tensor::new_1d(units));
		slot_index = s.height() - 1;
	}
	else if (sim_tensor[slot_index] <= MAX_SIM_FOR_ALLOC) {
		s.vec().push_back(tensor::new_1d(units));
		slot_index = s.height() - 1;
	}

	tensor& slot = s[slot_index];
	strong_interface->ctx.pop(slot);
	strong_interface->fwd();
	slot.pop(strong_interface->cty);
}

void rnc::bwd() {
	// NO IMPLEMENTATION, R LEARNING ONLY
}

tensor& rnc::fwd(tensor& a_x) {
	x.pop(a_x);
	fwd();
	return y;
}

tensor& rnc::bwd(tensor& a_y_grad) {
	y_grad.pop(a_y_grad);
	bwd();
	return x_grad;
}

void rnc::signal(tensor& a_y_des) {
	strong_interface->signal(a_y_des);
}

void rnc::cycle(tensor& a_x, tensor& a_y_des) {
	x.pop(a_x);
	fwd();
	signal(a_y_des);
	bwd();
}

void rnc::recur(function<void(model*)> a_func) {
	a_func(this);
	weak_interface->recur(a_func);
	strong_interface->recur(a_func);
}

void rnc::compile() {
	weak_interface->compile();
	strong_interface->compile();
	x.group_add(weak_interface->x);
	x.group_add(strong_interface->x);
	weak_interface->cty.group_add(weak_interface->ctx);
	strong_interface->y.group_add(y);
}