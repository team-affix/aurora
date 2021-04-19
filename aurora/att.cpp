#include "pch.h"
#include "att.h"

using aurora::models::att;

att::~att() {

}

att::att() {

}

void att::pmt_wise(function<void(ptr<param>&)> a_func) {

}

model* att::clone() {

}

model* att::clone(function<void(ptr<param>&)> a_func) {

}

void att::fwd() {

}

void att::bwd() {

}

tensor& att::fwd(tensor& a_x) {

}

tensor& att::bwd(tensor& a_y_grad) {

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