#include "pch.h"
#include "ntm_ts.h"

using aurora::models::ntm_ts;

ntm_ts::~ntm_ts() {

}

ntm_ts::ntm_ts() {

}

ntm_ts::ntm_ts(
	size_t a_memory_height,
	size_t a_memory_width,
	size_t a_num_reads,
	size_t a_num_writes,
	vector<size_t> a_head_h_dims, 
	vector<int> a_valid_shifts,
	function<void(ptr<param>&)> a_func) {

	memory_height = a_memory_height;
	memory_width = a_memory_width;

	for (int i = 0; i < a_num_reads; i++)
		readers.push_back(new ntm_reader(a_memory_height, a_memory_width, a_head_h_dims, a_valid_shifts, a_func));
	for (int i = 0; i < a_num_writes; i++)
		writers.push_back(new ntm_writer(a_memory_height, a_memory_width, a_head_h_dims, a_valid_shifts, a_func));

}

void ntm_ts::pmt_wise(function<void(ptr<param>&)> a_func) {
	for (ptr<ntm_reader>& m : readers)
		m->pmt_wise(a_func);
	for (ptr<ntm_writer>& m : writers)
		m->pmt_wise(a_func);
}

model* ntm_ts::clone() {
	ntm_ts* result = new ntm_ts();
	result->memory_height = memory_height;
	result->memory_width = memory_width;
	for (ptr<ntm_reader>& m : readers)
		result->readers.push_back((ntm_reader*)m->clone());
	for (ptr<ntm_writer>& m : writers)
		result->writers.push_back((ntm_writer*)m->clone());
	return result;
}

model* ntm_ts::clone(function<void(ptr<param>&)> a_func) {
	ntm_ts* result = new ntm_ts();
	result->memory_height = memory_height;
	result->memory_width = memory_width;
	for (ptr<ntm_reader>& m : readers)
		result->readers.push_back((ntm_reader*)m->clone(a_func));
	for (ptr<ntm_writer>& m : writers)
		result->writers.push_back((ntm_writer*)m->clone(a_func));
	return result;
}

void ntm_ts::fwd() {

	y.clear();

	for (ptr<ntm_reader>& m : readers) {
		m->fwd();
		y.add_1d(m->y, y);
	}

	for (ptr<ntm_writer>& m : writers)
		m->fwd();
	
}

void ntm_ts::bwd() {

	mx_grad.clear();
	x_grad.clear();

	for (int i = writers.size() - 1; i >= 0; i--) {
		ptr<ntm_writer>& m = writers[i];
		m->bwd();
		x_grad.add_1d(m->x_grad, x_grad);
	}

	for (int i = 0; i < readers.size(); i++) {
		ptr<ntm_reader>& m = readers[i];
		m->bwd();
		mx_grad.add_2d(m->mx_grad, mx_grad);
		x_grad.add_1d(m->x_grad, x_grad);
	}

}

tensor& ntm_ts::fwd(tensor& a_x) {
	x.pop(a_x);
	fwd();
	return y;
}

tensor& ntm_ts::bwd(tensor& a_y_grad) {
	y_grad.pop(a_y_grad);
	bwd();
	return x_grad;
}

void ntm_ts::signal(tensor& a_y_des) {
	y.sub_1d(a_y_des, y_grad);
}

void ntm_ts::cycle(tensor& a_x, tensor& a_y_des) {
	x.pop(a_x);
	fwd();
	signal(a_y_des);
	bwd();
}

void ntm_ts::recur(function<void(model*)> a_func) {
	for (ptr<ntm_reader>& m : readers)
		m->recur(a_func);
	for (ptr<ntm_writer>& m : writers)
		m->recur(a_func);
}

void ntm_ts::compile() {

	x = tensor::new_1d(memory_width);
	x_grad = tensor::new_1d(memory_width);
	y = tensor::new_1d(memory_width);
	y_grad = tensor::new_1d(memory_width);

	mx = tensor::new_2d(memory_height, memory_width);
	mx_grad = tensor::new_2d(memory_height, memory_width);
	my = tensor::new_2d(memory_height, memory_width);
	my_grad = tensor::new_2d(memory_height, memory_width);

	for (ptr<ntm_reader>& m : readers)
	for (ptr<ntm_writer>& m : writers)
		m->compile();

	for (int i = 0; i < readers.size(); i++) {
		ptr<ntm_reader>& m = readers[i];
		m->compile();
		m->x.group_join(x);
		m->y_grad.group_join(y_grad);
		m->mx.group_join(mx);
	}

	tensor* l_mx = &mx;
	tensor* l_mx_grad = &mx_grad;
	for (int i = 0; i < writers.size(); i++) {
		ptr<ntm_writer>& m = writers[i];
		m->compile();
		m->x.group_join(x);
		m->mx.group_join(*l_mx);
		m->mx_grad.group_join(*l_mx_grad);
		l_mx = &m->y;
		l_mx_grad = &m->y_grad;
	}
	my.group_join(*l_mx);
	my_grad.group_join(*l_mx_grad);

}
