#include "pch.h"
#include "tensor.h"

using namespace aurora;
using namespace maths;

double& tensor::val() {
	return *val_ptr;
}

vector<tensor>& tensor::vec() {
	return *vec_ptr;
}

tensor& tensor::group_head() {
	if (group_prev_ptr == nullptr)
		return *this;
	else
		return group_prev_ptr->group_head();
}

tensor& tensor::group_tail() {
	if (group_next_ptr == nullptr)
		return *this;
	else
		return group_next_ptr->group_tail();
}

size_t tensor::group_size() {
	size_t result = 0;
	group_recur([&](tensor* elem) { result += 1; });
	return result;
}

tensor::~tensor() {
	group_leave();
}

tensor::tensor() {

}

tensor::tensor(double a_val) {
	this->val() = a_val;
}

tensor::tensor(vector<tensor> a_vec) {
	std::copy(a_vec.begin(), a_vec.end(), back_inserter(vec()));
}

tensor::tensor(initializer_list<tensor> a_il) {
	std::copy(a_il.begin(), a_il.end(), back_inserter(vec()));
}

void tensor::set(tensor a_other) {
	val() = a_other.val();
	resize(a_other.size());
	for (size_t i = 0; i < vec().size(); i++)
		at(i).set(a_other.at(i));
}

void tensor::pop(tensor a_other) {
	val() = a_other.val();
	for (size_t i = 0; i < a_other.vec().size(); i++)
		at(i).pop(a_other.at(i));
}

void tensor::ref_set(tensor& a_other) {
	val() = a_other.val();
	resize(a_other.size());
	for (size_t i = 0; i < vec().size(); i++)
		at(i).ref_set(a_other.at(i));
}

void tensor::ref_pop(tensor& a_other) {
	val() = a_other.val();
	for (size_t i = 0; i < vec().size(); i++)
		at(i).ref_pop(a_other.at(i));
}

void tensor::resize(size_t a_size) {
	vec().resize(a_size);
}

tensor tensor::new_1d(size_t a_a) {
	return new_1d(a_a, 0);
}

tensor tensor::new_1d(size_t a_a, tensor a_val) {
	tensor result;
	result.vec().resize(a_a);
	for (int i = 0; i < a_a; i++) {
		result[i] = a_val.clone();
	}
	return result;
}

tensor tensor::new_1d(size_t a_a, uniform_real_distribution<double>& a_urd, default_random_engine& a_re) {
	tensor result = new_1d(a_a);
	for (int i = 0; i < a_a; i++)
		result[i].val() = a_urd(a_re);
	return result;
}

tensor tensor::new_2d(size_t a_a, size_t a_b) {
	return new_2d(a_a, a_b, 0);
}

tensor tensor::new_2d(size_t a_a, size_t a_b, tensor a_val) {
	tensor result;
	result.vec().resize(a_a);
	for (int i = 0; i < a_a; i++)
		result[i] = new_1d(a_b, a_val);

	return result;
}

tensor tensor::new_2d(size_t a_a, size_t a_b, uniform_real_distribution<double>& a_urd, default_random_engine& a_re) {
	tensor result = new_1d(a_a);
	for (int i = 0; i < a_a; i++)
		result[i] = new_1d(a_b, a_urd, a_re);
	return result;
}

void tensor::abs_1d(tensor& a_output) {
	for (int i = 0; i < vec().size(); i++)
		a_output.at(i).val() = abs(at(i).val());
}

void tensor::abs_2d(tensor& a_output) {
	for (int i = 0; i < vec().size(); i++)
		at(i).abs_1d(a_output.at(i));
}

void tensor::sum_1d(tensor& a_output) {
	for (int i = 0; i < vec().size(); i++)
		a_output.val() += vec().at(i).val();
}

void tensor::sum_2d(tensor& a_output) {
	a_output.clear();
	for (int i = 0; i < vec().size(); i++)
		a_output.add_1d(at(i), a_output);
}

void tensor::tanh_1d(tensor& a_output) {
	for (int i = 0; i < size(); i++)
		a_output[i].val() = tanh(at(i).val());
}

void tensor::tanh_2d(tensor& a_output) {
	for (int i = 0; i < size(); i++)
		at(i).tanh_1d(a_output.at(i));
}

tensor tensor::mag_1d() {
	double sum = 0;
	for (int i = 0; i < size(); i++)
		sum += pow(at(i), 2);
	return sqrt(sum);
}

void tensor::zero_1d() {
	for (int i = 0; i < size(); i++)
		at(i).val() = 0;
}

void tensor::zero_2d() {
	for (int i = 0; i < size(); i++)
		at(i).zero_1d();
}

void tensor::zero() {
	if (size() == 0)
		val() = 0;
	else
		for (int i = 0; i < size(); i++)
			at(i).zero();
}

tensor tensor::abs_1d() {
	tensor result = new_1d(size());
	abs_1d(result);
	return result;
}

tensor tensor::abs_2d() {
	tensor result = new_2d(height(), width());
	abs_2d(result);
	return result;
}

tensor tensor::sum_1d() {
	tensor result = 0;
	sum_1d(result);
	return result;
}

tensor tensor::sum_2d() {
	tensor result = tensor::new_1d(width());
	sum_2d(result);
	return result;
}

tensor tensor::tanh_1d() {
	tensor result = tensor::new_1d(size());
	tanh_1d(result);
	return result;
}

tensor tensor::tanh_2d() {
	tensor result = tensor::new_2d(height(), width());
	tanh_2d(result);
	return result;
}

tensor tensor::up_rank(size_t a_n) {
	assert(a_n >= 0);
	if (a_n == 0)
		return *this;
	else
		return up_rank().up_rank(a_n - 1);
}

tensor tensor::up_rank() {
	return tensor({ *this });
}

tensor tensor::down_rank(size_t a_n) {
	assert(a_n >= 0);
	if (a_n == 0)
		return *this;
	else
		return down_rank().down_rank(a_n - 1);
}

tensor tensor::down_rank() {
	assert(vec().size() == 1);
	return vec().at(0);
}

tensor tensor::unroll() {
	size_t w = width();
	size_t h = height();
	tensor result = new_1d(w * h);
	for (int i = 0; i < h; i++)
		for (size_t j = 0; j < w; j++)
			result[i * w + j].group_join_all_ranks(at(i)[j]);
	return result;
}

tensor tensor::roll(size_t a_width) {
	size_t s = vec().size();
	assert(s % a_width == 0);
	size_t h = s / a_width;
	tensor result = new_2d(h, a_width);
	for (int i = 0; i < vec().size(); i++)
		result[i / a_width][i % a_width].group_join_all_ranks(at(i));
	return result;
}

size_t tensor::width() {
	return vec().at(0).size();
}

size_t tensor::height() {
	return size();
}

tensor tensor::max() {
	double result = -INFINITY;
	for (int i = 0; i < size(); i++)
		if (at(i) > result)
			result = at(i);
	return result;
}

tensor tensor::min() {
	double result = INFINITY;
	for (int i = 0; i < size(); i++)
		if (at(i) < result)
			result = at(i);
	return result;
}

int tensor::arg_max() {
	double val = -INFINITY;
	int result = -1;
	for (int i = 0; i < size(); i++)
		if (at(i) > val) {
			val = at(i);
			result = i;
		}
	return result;
}

int tensor::arg_min() {
	double val = INFINITY;
	int result = -1;
	for (int i = 0; i < size(); i++)
		if (at(i) < val) {
			val = at(i);
			result = i;
		}
	return result;
}

tensor tensor::row(size_t a_a) {
	tensor result;
	result.group_join_all_ranks(at(a_a));
	return result;
}

tensor tensor::col(size_t a_a) {
	tensor result = new_1d(vec().size());
	for (int i = 0; i < size(); i++)
		result[i].group_join_all_ranks(at(i)[a_a]);
	return result;
}

tensor tensor::range(size_t a_start, size_t a_len) {
	tensor result = new_1d(a_len);
	for (int i = 0; i < a_len; i++)
	{
		size_t src = a_start + i;
		size_t dst = i;
		result[dst].group_join_all_ranks(at(src));
	}
	return result;
}

tensor tensor::range_2d(size_t a_row, size_t a_col, size_t a_height, size_t a_width) {
	tensor result = new_2d(a_height, a_width);
	for (int i = 0; i < a_height; i++)
	{
		size_t src = a_row + i;
		size_t dst = i;
		tensor row_section = at(src).range(a_col, a_width);
		result[dst].group_join_all_ranks(row_section);
	}
	return result;
}

tensor tensor::clone_row(size_t a_a) {
	return at(a_a).clone();
}

tensor tensor::clone_col(size_t a_a) {
	return col(a_a).clone();
}

tensor tensor::clone_range(size_t a_start, size_t a_len) {
	tensor result = new_1d(a_len);
	for (int i = 0; i < a_len; i++)
	{
		size_t src = a_start + i;
		size_t dst = i;
		result[dst] = vec().at(src);
	}
	return result.clone();
}

tensor tensor::add_1d(tensor a_other) {
	tensor result = new_1d(vec().size());
	add_1d(a_other, result);
	return result;
}

void tensor::add_1d(tensor a_other, tensor& a_output) {
	assert(vec().size() == a_other.size());
	for (int i = 0; i < vec().size(); i++)
		a_output[i].val() = vec().at(i).val() + a_other.vec().at(i).val();
}

tensor tensor::sub_1d(tensor a_other) {
	tensor result = new_1d(vec().size());
	sub_1d(a_other, result);
	return result;
}

void tensor::sub_1d(tensor a_other, tensor& a_output) {
	assert(vec().size() == a_other.vec().size());
	for (int i = 0; i < vec().size(); i++)
		a_output[i].val() = vec().at(i).val() - a_other.vec().at(i).val();
}

tensor tensor::mul_1d(tensor a_other) {
	tensor result = new_1d(vec().size());
	mul_1d(a_other, result);
	return result;
}

void tensor::mul_1d(tensor a_other, tensor& a_output) {
	assert(vec().size() == a_other.vec().size());
	for (int i = 0; i < vec().size(); i++)
		a_output[i].val() = vec().at(i).val() * a_other.vec().at(i).val();
}

tensor tensor::div_1d(tensor a_other) {
	tensor result = new_1d(vec().size());
	div_1d(a_other, result);
	return result;
}

void tensor::div_1d(tensor a_other, tensor& a_output) {
	assert(vec().size() == a_other.vec().size());
	for (int i = 0; i < vec().size(); i++)
		a_output[i].val() = vec().at(i).val() / a_other.vec().at(i).val();
}

tensor tensor::dot_1d(tensor a_other) {
	tensor result;
	dot_1d(a_other, result);
	return result;
}

void tensor::dot_1d(tensor a_other, tensor& a_output) {
	assert(vec().size() == a_other.vec().size());
	a_output.val() = (double)0;
	for (int i = 0; i < vec().size(); i++)
		a_output.val() += vec().at(i).val() * a_other.vec().at(i).val();
}

tensor tensor::add_2d(tensor a_other) {
	tensor result = new_2d(vec().size(), vec().at(0).vec().size());
	add_2d(a_other, result);
	return result;
}

void tensor::add_2d(tensor a_other, tensor& a_output) {
	assert(vec().size() == a_other.vec().size());
	for (int i = 0; i < vec().size(); i++)
		at(i).add_1d(a_other.at(i), a_output.at(i));
}

tensor tensor::sub_2d(tensor a_other) {
	tensor result = new_2d(vec().size(), vec().at(0).size());
	sub_2d(a_other, result);
	return result;
}

void tensor::sub_2d(tensor a_other, tensor& a_output) {
	assert(vec().size() == a_other.vec().size());
	for (int i = 0; i < vec().size(); i++)
		at(i).sub_1d(a_other.at(i), a_output.at(i));
}

tensor tensor::mul_2d(tensor a_other) {
	tensor result = new_2d(vec().size(), vec().at(0).size());
	mul_2d(a_other, result);
	return result;
}

void tensor::mul_2d(tensor a_other, tensor& a_output) {
	assert(vec().size() == a_other.vec().size());
	for (int i = 0; i < vec().size(); i++)
		at(i).mul_1d(a_other.at(i), a_output.at(i));
}

tensor tensor::div_2d(tensor a_other) {
	tensor result = new_2d(vec().size(), vec().at(0).size());
	div_2d(a_other, result);
	return result;
}

void tensor::div_2d(tensor a_other, tensor& a_output) {
	assert(vec().size() == a_other.vec().size());
	for (int i = 0; i < vec().size(); i++)
		at(i).div_1d(a_other.at(i), a_output.at(i));
}

tensor tensor::dot_2d(tensor a_other) {
	tensor result = new_2d(height(), a_other.width());
	dot_2d(a_other, result);
	return result;
}

void tensor::dot_2d(tensor a_other, tensor& a_output) {
	assert(width() == a_other.height());
	for (int i = 0; i < height(); i++)
		for (size_t j = 0; j < a_other.width(); j++)
		row(i).dot_1d(a_other.col(j), a_output[i][j]);
}

void tensor::concat(tensor& a_other, tensor& a_output) {
	for (int i = 0; i < size(); i++)
		a_output[i].group_join_all_ranks(at(i));
	for (int i = 0; i < a_other.size(); i++)
		a_output[i + size()].group_join_all_ranks(a_other[i]);
}

double tensor::cos_sim(tensor& a_other) {
	double dot = abs(dot_1d(a_other));
	double mag_a = mag_1d();
	double mag_b = a_other.mag_1d();
	return dot / (mag_a * mag_b);
}

tensor tensor::concat(tensor& a_other) {
	tensor result = tensor::new_1d(size() + a_other.size());
	concat(a_other, result);
	return result;
}

void tensor::link(tensor& a_other) {
	group_recur([&](tensor* elem) {
		elem->val_ptr.link(a_other.val_ptr);
		elem->resize(a_other.size());
		for (int i = 0; i < a_other.size(); i++)
			elem->at(i).link(a_other.at(i));
	});
}

void tensor::unlink() {
	val_ptr = nullptr;
	for (int i = 0; i < size(); i++)
		at(i).unlink();
}

void tensor::rank_recur(function<void(tensor*)> a_func) {
	if (size() != 0)
		for (int i = 0; i < size(); i++)
			at(i).rank_recur(a_func);
	a_func(this);
}

void tensor::group_recur_fwd(function<void(tensor*)> a_func) {
	if (group_next_ptr != nullptr)
		group_next_ptr->group_recur_fwd(a_func);
	a_func(this);
}

void tensor::group_recur_bwd(function<void(tensor*)> a_func) {
	if (group_prev_ptr != nullptr)
		group_prev_ptr->group_recur_bwd(a_func);
	a_func(this);
}

void tensor::group_recur(function<void(tensor*)> a_func) {
	if (group_prev_ptr != nullptr)
		group_prev_ptr->group_recur_bwd(a_func);
	if (group_next_ptr != nullptr)
		group_next_ptr->group_recur_fwd(a_func);
	a_func(this);
}

bool tensor::group_contains(tensor* a_ptr) {
	bool result = false;
	group_recur([&](tensor* tens) {
		if (tens == a_ptr)
			result = true;
	});
	return result;
}

void tensor::group_add(tensor& a_other) {
	a_other.group_join(*this);
}

void tensor::group_remove(tensor& a_other) {
	a_other.group_leave();
}

void tensor::group_join(tensor& a_other) {
	link(a_other); // LINKS ENTIRE GROUP TO a_other's GROUP
	group_recur([&](tensor* elem) {
		// PREVENT ADDING NODES TO SAME GROUP TWICE
		if (!a_other.group_contains(elem)) {
			tensor& l_group_tail = a_other.group_tail();
			l_group_tail.group_next_ptr = elem;
			elem->group_prev_ptr = &l_group_tail;
			elem->group_next_ptr = nullptr;
		}
	});
}

void tensor::group_leave() {
	if (group_prev_ptr != nullptr)
		group_prev_ptr->group_next_ptr = group_next_ptr;
	if (group_next_ptr != nullptr)
		group_next_ptr->group_prev_ptr = group_prev_ptr;
	group_prev_ptr = nullptr;
	group_next_ptr = nullptr;
}

void tensor::group_disband() {
	group_recur([](tensor* elem) { elem->group_leave(); });
}

void tensor::group_add_all_ranks(tensor& a_other) {
	a_other.group_join_all_ranks(*this);
}

void tensor::group_remove_all_ranks(tensor& a_other) {
	a_other.group_leave_all_ranks();
}

void tensor::group_join_all_ranks(tensor& a_other) {
	link(a_other); // LINKS ENTIRE GROUP TO a_other's GROUP
	group_recur([&](tensor* elem) {

		for (int i = 0; i < size(); i++)
			at(i).group_join_all_ranks(a_other.at(i));

		group_join(a_other);
		
		// PREVENT ADDING NODES TO SAME GROUP TWICE
		if (!a_other.group_contains(elem)) {
			tensor& l_group_tail = a_other.group_tail();
			l_group_tail.group_next_ptr = elem;
			elem->group_prev_ptr = &l_group_tail;
			elem->group_next_ptr = nullptr;
		}

	});
}

void tensor::group_leave_all_ranks() {
	for (int i = 0; i < size(); i++)
		at(i).group_leave_all_ranks();
	group_leave();
}

void tensor::group_disband_all_ranks() {
	for (int i = 0; i < size(); i++)
		at(i).group_disband_all_ranks();
	group_disband();
}

tensor tensor::clone() {
	tensor result = tensor(val());
	result.vec().resize(vec().size());
	for (int i = 0; i < vec().size(); i++)
		result.at(i) = at(i).clone();
	return result;
}

tensor tensor::link() {
	tensor result = tensor();
	result.link(*this);
	return result;
}

string tensor::to_string() {
	if (vec().size() == 0)
		return std::to_string(val());
	string result = "[";
	for (int i = 0; i < vec().size() - 1; i++)
		result += vec().at(i).to_string() + " ";
	result += vec().at(vec().size() - 1).to_string() + "]";
	return result;
}

void tensor::clear() {
	val() = (double)0;
	for (int i = 0; i < vec().size(); i++)
		vec().at(i).clear();
}

size_t tensor::size() {
	return vec().size();
}

tensor& tensor::at(size_t a_a) {
	return vec().at(a_a);
}

tensor::operator double& () {
	return val();
}

tensor& tensor::operator[](size_t a) {
	return vec()[a];
}