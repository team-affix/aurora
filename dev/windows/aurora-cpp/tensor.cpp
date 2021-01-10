#include "tensor.h"
#include <assert.h>

using namespace aurora;
using namespace math;

tensor::tensor() {

}

tensor::tensor(double _val) {
	this->val = _val;
}

tensor::tensor(vector<tensor> _vec) {
	std::copy(_vec.begin(), _vec.end(), back_inserter(*this));
}

tensor::tensor(initializer_list<tensor> _il) {
	std::copy(_il.begin(), _il.end(), back_inserter(*this));
}

void tensor::set(tensor _other) {
	_other.clone(*this);
}

tensor tensor::new_1d(size_t _a) {
	tensor result;
	result.resize(_a);
	for (int i = 0; i < _a; i++) {
		result[i] = 0;
	}
	return result;
}

tensor tensor::new_2d(size_t _a, size_t _b) {
	tensor result;
	result.resize(_a);
	for (int i = 0; i < _a; i++) 
		result[i] = new_1d(_b);
	
	return result;
}

tensor tensor::sum_1d() {
	tensor result = 0;
	for (int i = 0; i < size(); i++) 
		result.val += at(i).val;
	
	return result;
}

tensor tensor::up_rank(size_t n) {
	assert(n >= 0);
	if (n == 0)
		return *this;
	else
		return up_rank().up_rank(n - 1);
}

tensor tensor::up_rank() {
	return tensor({ *this });
}

tensor tensor::down_rank(size_t n) {
	assert(n >= 0);
	if (n == 0)
		return *this;
	else
		return down_rank().down_rank(n - 1);
}

tensor tensor::down_rank() {
	assert(size() == 1);
	return at(0);
}

tensor tensor::unroll() {
	size_t w = width();
	size_t h = height();
	tensor result = new_1d(w * h);
	for (int i = 0; i < h; i++)
		for (size_t j = 0; j < w; j++)
			result[i * w + j] = at(i)[j];
	return result;
}

tensor tensor::roll(size_t _width) {
	size_t s = size();
	assert(s % _width == 0);
	size_t h = s / _width;
	tensor result = new_2d(h, _width);
	for (int i = 0; i < size(); i++)
		result[i / _width][i % _width] = at(i);
	return result;
}

size_t tensor::width() {
	return at(0).size();
}

size_t tensor::height() {
	return size();
}

tensor tensor::row(size_t _a) {
	tensor result = at(_a);
	return result;
}

tensor tensor::col(size_t _a) {
	tensor result = new_1d(size());
	for (int i = 0; i < size(); i++)
		result[i] = at(i)[_a];
	return result;
}

tensor tensor::range(size_t _start, size_t _len) {
	tensor result = new_1d(_len);
	for (int i = 0; i < _len; i++)
	{
		size_t src = _start + i;
		size_t dst = i;
		result[dst] = at(src);
	}
	return result;
}

tensor tensor::clone_row(size_t _a) {
	tensor result = at(_a);
	return result.clone();
}

tensor tensor::clone_col(size_t _a) {
	tensor result = new_1d(size());
	for (int i = 0; i < size(); i++)
		result[i] = at(i)[_a];
	return result.clone();
}

tensor tensor::clone_range(size_t _start, size_t _len) {
	tensor result = new_1d(_len);
	for (int i = 0; i < _len; i++)
	{
		size_t src = _start + i;
		size_t dst = i;
		result[dst] = at(src);
	}
	return result.clone();
}

tensor tensor::add_1d(tensor _other) {
	tensor result = new_1d(size());
	add_1d(_other, result);
	return result;
}

void tensor::add_1d(tensor _other, tensor& _output) {
	assert(size() == _other.size());
	for (int i = 0; i < size(); i++)
		_output[i] = at(i).val + _other.at(i).val;
}

tensor tensor::sub_1d(tensor _other) {
	tensor result = new_1d(size());
	sub_1d(_other, result);
	return result;
}

void tensor::sub_1d(tensor _other, tensor& _output) {
	assert(size() == _other.size());
	for (int i = 0; i < size(); i++)
		_output[i] = at(i).val - _other.at(i).val;
}

tensor tensor::mul_1d(tensor _other) {
	tensor result = new_1d(size());
	mul_1d(_other, result);
	return result;
}

void tensor::mul_1d(tensor _other, tensor& _output) {
	assert(size() == _other.size());
	for (int i = 0; i < size(); i++)
		_output[i] = at(i).val * _other.at(i).val;
}

tensor tensor::div_1d(tensor _other) {
	tensor result = new_1d(size());
	div_1d(_other, result);
	return result;
}

void tensor::div_1d(tensor _other, tensor& _output) {
	assert(size() == _other.size());
	for (int i = 0; i < size(); i++)
		_output[i] = at(i).val / _other.at(i).val;
}

tensor tensor::dot_1d(tensor _other) {
	tensor result;
	dot_1d(_other, result);
	return result;
}

void tensor::dot_1d(tensor _other, tensor& _output) {
	assert(size() == _other.size());
	_output.val = (double)0;
	for (int i = 0; i < size(); i++)
		_output.val += at(i).val * _other.at(i).val;
}

tensor tensor::add_2d(tensor _other) {
	tensor result = new_2d(size(), at(0).size());
	add_2d(_other, result);
	return result;
}

void tensor::add_2d(tensor _other, tensor& _output) {
	assert(size() == _other.size());
	for (int i = 0; i < size(); i++)
		_output[i] = at(i).add_1d(_other[i]);
}

tensor tensor::sub_2d(tensor _other) {
	tensor result = new_2d(size(), at(0).size());
	sub_2d(_other, result);
	return result;
}

void tensor::sub_2d(tensor _other, tensor& _output) {
	assert(size() == _other.size());
	for (int i = 0; i < size(); i++)
		_output[i] = at(i).sub_1d(_other[i]);
}

tensor tensor::mul_2d(tensor _other) {
	tensor result = new_2d(size(), at(0).size());
	mul_2d(_other, result);
	return result;
}

void tensor::mul_2d(tensor _other, tensor& _output) {
	assert(size() == _other.size());
	for (int i = 0; i < size(); i++)
		_output[i] = at(i).mul_1d(_other[i]);
}

tensor tensor::div_2d(tensor _other) {
	tensor result = new_2d(size(), at(0).size());
	div_2d(_other, result);
	return result;
}

void tensor::div_2d(tensor _other, tensor& _output) {
	assert(size() == _other.size());
	for (int i = 0; i < size(); i++)
		_output[i] = at(i).div_1d(_other[i]);
}

tensor tensor::dot_2d(tensor _other) {
	tensor result = new_2d(height(), _other.width());
	dot_2d(_other, result);
	return result;
}

void tensor::dot_2d(tensor _other, tensor& _output) {
	assert(width() == _other.height());
	for (int i = 0; i < height(); i++)
		for (size_t j = 0; j < _other.width(); j++)
			_output[i][j] = at(i).dot_1d(_other.col(j));
}

void tensor::clone(tensor& _output) {
	assert(size() == _output.size());
	_output.val = val;
	for (int i = 0; i < size(); i++)
		at(i).clone(_output.at(i));
}

void tensor::link(tensor& _other) {
	val_ptr.link(_other.val_ptr);
	val = val_ptr.get();
	for (int i = 0; i < size(); i++)
		at(i).link(_other.at(i));
}

void tensor::unlink() {
	val_ptr.unlink();
	for (int i = 0; i < size(); i++)
		at(i).unlink();
}

tensor tensor::clone() {
	tensor result = tensor(val);
	result.resize(size());
	for (int i = 0; i < size(); i++)
		result.at(i) = at(i).clone();
	return result;
}

string tensor::to_string() {
	if (size() == 0)
		return std::to_string(val);
	string result = "[";
	for (int i = 0; i < size() - 1; i++)
		result += at(i).to_string() + " ";
	result += at(size() - 1).to_string() + "]";
	return result;
}

void tensor::operator=(double _other) {
	val = _other;
}

void tensor::operator=(tensor _other) {
	*this = _other;
}

void tensor::clear() {
	val = (double)0;
	for (int i = 0; i < size(); i++)
		at(i).clear();
}

tensor::operator double& () {
	return val;
}