#include "tensor.h"
#include <assert.h>

using namespace aurora;
using namespace math;

double& tensor::val() {
	return *val_ptr.get();
}

vector<tensor>& tensor::vec() {
	return *vec_ptr.get();
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
	vec() = a_other.vec();
}

void tensor::pop(tensor a_other) {
	a_other.clone(*this);
}

tensor tensor::new_1d(size_t a_a) {
	tensor result;
	result.vec().resize(a_a);
	for (int i = 0; i < a_a; i++) {
		result[i] = 0;
	}
	return result;
}

tensor tensor::new_2d(size_t a_a, size_t a_b) {
	tensor result;
	result.vec().resize(a_a);
	for (int i = 0; i < a_a; i++) 
		result[i] = new_1d(a_b);
	
	return result;
}

tensor tensor::sum_1d() {
	tensor result = 0;
	for (int i = 0; i < vec().size(); i++) 
		result.val() += vec().at(i).val();
	
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
			result[i * w + j] = vec().at(i)[j];
	return result;
}

tensor tensor::roll(size_t a_width) {
	size_t s = vec().size();
	assert(s % a_width == 0);
	size_t h = s / a_width;
	tensor result = new_2d(h, a_width);
	for (int i = 0; i < vec().size(); i++)
		result[i / a_width][i % a_width] = vec().at(i);
	return result;
}

size_t tensor::width() {
	return vec().at(0).size();
}

size_t tensor::height() {
	return size();
}

tensor tensor::row(size_t a_a) {
	tensor result = vec().at(a_a);
	return result;
}

tensor tensor::col(size_t a_a) {
	tensor result = new_1d(vec().size());
	for (int i = 0; i < size(); i++)
		result[i] = vec().at(i)[a_a];
	return result;
}

tensor tensor::range(size_t a_start, size_t a_len) {
	tensor result = new_1d(a_len);
	for (int i = 0; i < a_len; i++)
	{
		size_t src = a_start + i;
		size_t dst = i;
		result[dst] = at(src);
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
		a_output[i] = vec().at(i).val() + a_other.vec().at(i).val();
}

tensor tensor::sub_1d(tensor a_other) {
	tensor result = new_1d(vec().size());
	sub_1d(a_other, result);
	return result;
}

void tensor::sub_1d(tensor a_other, tensor& a_output) {
	assert(vec().size() == a_other.vec().size());
	for (int i = 0; i < vec().size(); i++)
		a_output[i] = vec().at(i).val() - a_other.vec().at(i).val();
}

tensor tensor::mul_1d(tensor a_other) {
	tensor result = new_1d(vec().size());
	mul_1d(a_other, result);
	return result;
}

void tensor::mul_1d(tensor a_other, tensor& a_output) {
	assert(vec().size() == a_other.vec().size());
	for (int i = 0; i < vec().size(); i++)
		a_output[i] = vec().at(i).val() * a_other.vec().at(i).val();
}

tensor tensor::div_1d(tensor a_other) {
	tensor result = new_1d(vec().size());
	div_1d(a_other, result);
	return result;
}

void tensor::div_1d(tensor a_other, tensor& a_output) {
	assert(vec().size() == a_other.vec().size());
	for (int i = 0; i < vec().size(); i++)
		a_output[i] = vec().at(i).val() / a_other.vec().at(i).val();
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
		a_output[i] = vec().at(i).add_1d(a_other[i]);
}

tensor tensor::sub_2d(tensor a_other) {
	tensor result = new_2d(vec().size(), vec().at(0).size());
	sub_2d(a_other, result);
	return result;
}

void tensor::sub_2d(tensor a_other, tensor& a_output) {
	assert(vec().size() == a_other.vec().size());
	for (int i = 0; i < vec().size(); i++)
		a_output[i] = vec().at(i).sub_1d(a_other[i]);
}

tensor tensor::mul_2d(tensor a_other) {
	tensor result = new_2d(vec().size(), vec().at(0).size());
	mul_2d(a_other, result);
	return result;
}

void tensor::mul_2d(tensor a_other, tensor& a_output) {
	assert(vec().size() == a_other.vec().size());
	for (int i = 0; i < vec().size(); i++)
		a_output[i] = vec().at(i).mul_1d(a_other[i]);
}

tensor tensor::div_2d(tensor a_other) {
	tensor result = new_2d(vec().size(), vec().at(0).size());
	div_2d(a_other, result);
	return result;
}

void tensor::div_2d(tensor a_other, tensor& a_output) {
	assert(vec().size() == a_other.vec().size());
	for (int i = 0; i < vec().size(); i++)
		a_output[i] = vec().at(i).div_1d(a_other[i]);
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
			a_output[i][j] = vec().at(i).dot_1d(a_other.col(j));
}

void tensor::clone(tensor& a_output) {
	assert(vec().size() == a_output.vec().size());
	a_output.val() = val();
	for (int i = 0; i < vec().size(); i++)
		vec().at(i).clone(a_output.vec().at(i));
}

void tensor::link(tensor& a_other) {
	val_ptr.link(a_other.val_ptr);
	vec_ptr.link(a_other.vec_ptr);
}

void tensor::unlink() {
	val_ptr.unlink();
	vec_ptr.unlink();
	for (int i = 0; i < vec().size(); i++)
		vec().at(i).unlink();
}

tensor tensor::clone() {
	tensor result = tensor(val());
	result.vec().resize(vec().size());
	for (int i = 0; i < vec().size(); i++)
		result.at(i) = at(i).clone();
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