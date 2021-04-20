#include "pch.h"
#include "att.h"

using aurora::models::att;

att::~att() {

}

att::att() {

}

att::att(size_t a_units, vector<size_t> a_h_dims, function<void(ptr<param>&)> a_func) {
	this->units = a_units;
	att_ts_template = new att_ts(a_units, a_h_dims, a_func);
	models = new sync(att_ts_template.get());
	internal_lstm = new lstm(a_units, a_func);
}

void att::pmt_wise(function<void(ptr<param>&)> a_func) {

}

model* att::clone() {
	return 0;
}

model* att::clone(function<void(ptr<param>&)> a_func) {
	return 0;
}

void att::fwd() {

}

void att::bwd() {

}

tensor& att::fwd(tensor& a_x) {
	return y;
}

tensor& att::bwd(tensor& a_y_grad) {
	return x_grad;
}

void att::signal(tensor& a_y_des) {

}

void att::cycle(tensor& a_x, tensor& a_y_des) {

}

void att::recur(function<void(model*)> a_func) {

}

void att::compile() {

}

void att::prep(size_t a_n, size_t b_n) {

}

void att::unroll(size_t a_n, size_t b_n) {

}