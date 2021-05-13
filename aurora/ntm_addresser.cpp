#include "pch.h"
#include "ntm_addresser.h"

using aurora::models::ntm_addresser;

ntm_addresser::~ntm_addresser() {

}

ntm_addresser::ntm_addresser() {

}

void ntm_addresser::pmt_wise(function<void(ptr<param>&)> a_func) {

}

model* ntm_addresser::clone() {
	return nullptr;
}

model* ntm_addresser::clone(function<void(ptr<param>&)> a_func) {
	return nullptr;
}

void ntm_addresser::fwd() {

}

void ntm_addresser::bwd() {

}

tensor& ntm_addresser::fwd(tensor& a_x) {
	return y;
}

tensor& ntm_addresser::bwd(tensor& a_y_grad) {
	return x_grad;
}

void ntm_addresser::signal(tensor& a_y_des) {

}

void ntm_addresser::cycle(tensor& a_x, tensor& a_y_des) {

}

void ntm_addresser::recur(function<void(model*)> a_func) {

}

void ntm_addresser::compile() {

}