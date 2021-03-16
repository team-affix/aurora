#include "rnc.h"

using aurora::models::rnc;

rnc::rnc() {

}

rnc::rnc(
	size_t a_x_units,
	size_t a_y_units,
	size_t a_weak_units,
	size_t a_strong_units,
	function<void(ptr<param>&)> a_init) {

	this->x_units = a_x_units;
	this->y_units = a_y_units;
	this->weak_units = a_weak_units;
	this->strong_units = a_strong_units;

	this->weak_in = pseudo::tnn({ x_units, weak_units }, pseudo::nlr(0.3), a_init);
	this->weak_mid = new lstm_ts(weak_units, a_init);
	this->weak_out = pseudo::tnn({ weak_units, 1 }, pseudo::nlr(0.3), a_init);

	this->strong_in = pseudo::tnn({ x_units, strong_units }, pseudo::nlr(0.3), a_init);
	this->strong_mid = new lstm_ts(strong_units, a_init);
	this->strong_out = pseudo::tnn({ strong_units, y_units }, pseudo::nlr(0.3), a_init);

	this->weak_interface = new sequential({ weak_in.get(), weak_mid.get(), weak_out.get() });
	this->strong_interface = new sequential({ strong_in.get(), strong_mid.get(), strong_out.get() });

	this->strong_memory = tensor::new_2d(0, a_strong_units);
}

rnc::rnc(
	size_t a_x_units,
	size_t a_y_units,
	size_t a_weak_units,
	size_t a_strong_units,
	ptr<model> a_weak_in,
	ptr<lstm_ts> a_weak_mid,
	ptr<model> a_weak_out,
	ptr<model> a_strong_in,
	ptr<lstm_ts> a_strong_mid,
	ptr<model> a_strong_out,
	tensor a_strong_memory) {

	this->x_units = a_x_units;
	this->y_units = a_y_units;
	this->weak_units = a_weak_units;
	this->strong_units = a_strong_units;

	this->weak_in = a_weak_in;
	this->weak_mid = a_weak_mid;
	this->weak_out = a_weak_out;

	this->strong_in = a_strong_in;
	this->strong_mid = a_strong_mid;
	this->strong_out = a_strong_out;

	this->weak_interface = new sequential({ weak_in.get(), weak_mid.get(), weak_out.get() });
	this->strong_interface = new sequential({ strong_in.get(), strong_mid.get(), strong_out.get() });

	this->strong_memory = a_strong_memory;
}

void rnc::pmt_wise(function<void(ptr<param>&)> a_func) {
	weak_interface->pmt_wise(a_func);
	strong_interface->pmt_wise(a_func);
}

model* rnc::clone() {
	rnc* result = new rnc(
		x_units,
		y_units,
		weak_units,
		strong_units,
		weak_in->clone(),
		(lstm_ts*)weak_mid->clone(),
		weak_out->clone(),
		strong_in->clone(),
		(lstm_ts*)strong_mid->clone(),
		strong_out->clone(),
		strong_memory.clone()
	);
	return result;
}

model* rnc::clone(function<void(ptr<param>&)> a_init) {
	rnc* result = new rnc(
		x_units,
		y_units,
		weak_units,
		strong_units,
		weak_in->clone(a_init),
		(lstm_ts*)weak_mid->clone(a_init),
		weak_out->clone(a_init),
		strong_in->clone(a_init),
		(lstm_ts*)strong_mid->clone(a_init),
		strong_out->clone(a_init),
		strong_memory.clone()
	); 
	return result;
}

void rnc::fwd() {
	// TREAT X AS TENSOR 2D
	weak_interface->fwd();

	int slot_index = weak_interface->y[0] * 100;

	if (slot_index < 0 || slot_index >= strong_memory.height()) {
		strong_memory.vec().push_back(tensor::new_1d(strong_units));
		slot_index = strong_memory.height() - 1;
	}

	tensor& slot = strong_memory[slot_index];
	strong_mid->ctx.pop(slot);
	strong_interface->fwd();
	slot.pop(strong_mid->cty);
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
	x.group_join(weak_interface->x);
	x.group_join(strong_interface->x);
	y.group_join(strong_interface->y);
	y_grad.group_join(strong_interface->y_grad);
}