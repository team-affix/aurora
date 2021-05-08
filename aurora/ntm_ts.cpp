#include "pch.h"
#include "ntm_ts.h"

using aurora::models::ntm_ts;

ntm_ts::~ntm_ts() {

}

ntm_ts::ntm_ts() {

}

void ntm_ts::pmt_wise(function<void(ptr<param>&)> a_func) {

}

model* ntm_ts::clone() {
	
}

model* ntm_ts::clone(function<void(ptr<param>&)> a_func) {

}

void ntm_ts::fwd() {

}

void ntm_ts::bwd() {

}

tensor& ntm_ts::fwd(tensor& a_x) {

}

tensor& ntm_ts::bwd(tensor& a_y_grad) {

}

void ntm_ts::signal(tensor& a_y_des) {

}

void ntm_ts::cycle(tensor& a_x, tensor& a_y_des) {

}

void ntm_ts::recur(function<void(model*)> a_func) {

}

void ntm_ts::compile() {

}