#include "pch.h"
#include "ntm_location_addresser.h"

using aurora::models::ntm_location_addresser;

ntm_location_addresser::~ntm_location_addresser() {

}

ntm_location_addresser::ntm_location_addresser() {

}

ntm_location_addresser::ntm_location_addresser(size_t a_memory_height, vector<int> a_valid_shifts) {
	memory_height = a_memory_height;
	shift_units = a_valid_shifts.size();
	internal_interpolate = new interpolate(a_memory_height);
	internal_shift = new shift(a_memory_height, a_valid_shifts);
	internal_power = new power(a_memory_height);
	internal_normalize = new normalize(a_memory_height);
}

void ntm_location_addresser::pmt_wise(function<void(ptr<param>&)> a_func) {
	internal_interpolate->pmt_wise(a_func);
	internal_shift->pmt_wise(a_func);
	internal_power->pmt_wise(a_func);
	internal_normalize->pmt_wise(a_func);
}

model* ntm_location_addresser::clone() {
	ntm_location_addresser* result = new ntm_location_addresser();
	result->memory_height = memory_height;
	result->shift_units = shift_units;
	result->internal_interpolate = (interpolate*)internal_interpolate->clone();
	result->internal_shift = (shift*)internal_shift->clone();
	result->internal_power = (power*)internal_power->clone();
	result->internal_normalize = (normalize*)internal_normalize->clone();
	return result;
}

model* ntm_location_addresser::clone(function<void(ptr<param>&)> a_func) {
	ntm_location_addresser* result = new ntm_location_addresser();
	result->memory_height = memory_height;
	result->shift_units = shift_units;
	result->internal_interpolate = (interpolate*)internal_interpolate->clone(a_func);
	result->internal_shift = (shift*)internal_shift->clone(a_func);
	result->internal_power = (power*)internal_power->clone(a_func);
	result->internal_normalize = (normalize*)internal_normalize->clone(a_func);
	return result;
}

void ntm_location_addresser::fwd() {
	internal_interpolate->fwd();
	internal_shift->fwd();
	internal_power->fwd();
	internal_normalize->fwd();
}

void ntm_location_addresser::bwd() {
	y_grad.add_1d(wy_grad, internal_normalize->y_grad);
	internal_normalize->bwd();
	internal_power->bwd();
	internal_shift->bwd();
	internal_interpolate->bwd();
}

tensor& ntm_location_addresser::fwd(tensor& a_x) {
	x.pop(a_x);
	fwd();
	return y;
}

tensor& ntm_location_addresser::bwd(tensor& a_y_grad) {
	y_grad.pop(a_y_grad);
	bwd();
	return x_grad;
}

void ntm_location_addresser::signal(tensor& a_y_des) {
	y.sub_1d(a_y_des, y_grad);
}

void ntm_location_addresser::cycle(tensor& a_x, tensor& a_y_des) {
	x.pop(a_x);
	fwd();
	signal(a_y_des);
	bwd();
}

void ntm_location_addresser::recur(function<void(model*)> a_func) {
	a_func(this);
	internal_interpolate->recur(a_func);
	internal_shift->recur(a_func);
	internal_power->recur(a_func);
	internal_normalize->recur(a_func);
}

void ntm_location_addresser::compile() {
	x = tensor::new_1d(memory_height);
	x_grad = tensor::new_1d(memory_height);
	y = tensor::new_1d(memory_height);
	y_grad = tensor::new_1d(memory_height);
	wx = tensor::new_1d(memory_height);
	wx_grad = tensor::new_1d(memory_height);
	wy = tensor::new_1d(memory_height);
	wy_grad = tensor::new_1d(memory_height);
	g = tensor::new_1d(1);
	g_grad = tensor::new_1d(1);
	s = tensor::new_1d(shift_units);
	s_grad = tensor::new_1d(shift_units);
	gamma = tensor::new_1d(1);
	gamma_grad = tensor::new_1d(1);

	internal_interpolate->compile();
	internal_shift->compile();
	internal_power->compile();
	internal_normalize->compile();
	
	g.group_join(internal_interpolate->amount);
	g_grad.group_join(internal_interpolate->amount_grad);
	s.group_join(internal_shift->amount);
	s_grad.group_join(internal_shift->amount_grad);
	gamma.group_join(internal_power->amount);
	gamma_grad.group_join(internal_power->amount_grad);

	wx.group_join(internal_interpolate->x[0]);
	wx_grad.group_join(internal_interpolate->x_grad[0]);
	x.group_join(internal_interpolate->x[1]);
	x_grad.group_join(internal_interpolate->x_grad[1]);
	internal_interpolate->y.group_join(internal_shift->x);
	internal_interpolate->y_grad.group_join(internal_shift->x_grad);
	internal_shift->y.group_join(internal_power->x);
	internal_shift->y_grad.group_join(internal_power->x_grad);
	internal_power->y.group_join(internal_normalize->x);
	internal_power->y_grad.group_join(internal_normalize->x_grad);
	internal_normalize->y.group_join(y);
	y.group_join(wy);

}