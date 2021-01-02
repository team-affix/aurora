#include "complex.h"
#include <assert.h>

using namespace aurora;
using namespace math;

complex::complex() {

}

complex::complex(double _val) {
	this->val.val() = _val;
}

complex::complex(vector<complex> _vec) {
	std::copy(_vec.begin(), _vec.end(), back_inserter(*this));
}

complex::complex(initializer_list<complex> _il) {
	std::copy(_il.begin(), _il.end(), back_inserter(*this));
}

void complex::set(complex _other) {
	_other.clone(*this);
}

complex complex::new_1d(size_t _a) {
	complex result;
	result.resize(_a);
	for (size_t i = 0; i < _a; i++) {
		result[i] = 0;
	}
	return result;
}

complex complex::new_2d(size_t _a, size_t _b) {
	complex result;
	result.resize(_a);
	for (size_t i = 0; i < _a; i++) 
		result[i] = new_1d(_b);
	
	return result;
}

complex complex::sum_1d() {
	complex result = 0;
	for (size_t i = 0; i < size(); i++) 
		result.val.val() += at(i).val.val();
	
	return result;
}

complex complex::up_rank(size_t n) {
	assert(n >= 0);
	if (n == 0)
		return *this;
	else
		return up_rank().up_rank(n - 1);
}

complex complex::up_rank() {
	return complex({ *this });
}

complex complex::down_rank(size_t n) {
	assert(n >= 0);
	if (n == 0)
		return *this;
	else
		return down_rank().down_rank(n - 1);
}

complex complex::down_rank() {
	assert(size() == 1);
	return at(0);
}

size_t complex::width() {
	return at(0).size();
}

size_t complex::height() {
	return size();
}

complex complex::row(size_t _a) {
	complex result = at(_a);
	return result;
}

complex complex::col(size_t _a) {
	complex result = new_1d(size());
	for (size_t i = 0; i < size(); i++)
		result[i] = at(i)[_a];
	return result;
}

complex complex::range(size_t _start, size_t _len) {
	complex result = new_1d(_len);
	for (size_t i = 0; i < _len; i++)
	{
		size_t src = _start + i;
		size_t dst = i;
		result[dst] = at(src);
	}
	return result;
}

complex complex::clone_row(size_t _a) {
	complex result = at(_a);
	return result.clone();
}

complex complex::clone_col(size_t _a) {
	complex result = new_1d(size());
	for (size_t i = 0; i < size(); i++)
		result[i] = at(i)[_a];
	return result.clone();
}

complex complex::clone_range(size_t _start, size_t _len) {
	complex result = new_1d(_len);
	for (size_t i = 0; i < _len; i++)
	{
		size_t src = _start + i;
		size_t dst = i;
		result[dst] = at(src);
	}
	return result.clone();
}

complex complex::add_1d(complex _other) {
	complex result = new_1d(size());
	add_1d(_other, result);
	return result;
}

void complex::add_1d(complex _other, complex& _output) {
	assert(size() == _other.size());
	for (size_t i = 0; i < size(); i++)
		_output[i] = at(i).val.val() + _other.at(i).val.val();
}

complex complex::sub_1d(complex _other) {
	complex result = new_1d(size());
	sub_1d(_other, result);
	return result;
}

void complex::sub_1d(complex _other, complex& _output) {
	assert(size() == _other.size());
	for (size_t i = 0; i < size(); i++)
		_output[i] = at(i).val.val() - _other.at(i).val.val();
}

complex complex::mul_1d(complex _other) {
	complex result = new_1d(size());
	mul_1d(_other, result);
	return result;
}

void complex::mul_1d(complex _other, complex& _output) {
	assert(size() == _other.size());
	for (size_t i = 0; i < size(); i++)
		_output[i] = at(i).val.val() * _other.at(i).val.val();
}

complex complex::div_1d(complex _other) {
	complex result = new_1d(size());
	div_1d(_other, result);
	return result;
}

void complex::div_1d(complex _other, complex& _output) {
	assert(size() == _other.size());
	for (size_t i = 0; i < size(); i++)
		_output[i] = at(i).val.val() / _other.at(i).val.val();
}

complex complex::dot_1d(complex _other) {
	complex result;
	dot_1d(_other, result);
	return result;
}

void complex::dot_1d(complex _other, complex& _output) {
	assert(size() == _other.size());
	_output.val.val() = 0;
	for (size_t i = 0; i < size(); i++)
		_output.val.val() += at(i).val.val() * _other.at(i).val.val();
}

complex complex::add_2d(complex _other) {
	complex result = new_2d(size(), at(0).size());
	add_2d(_other, result);
	return result;
}

void complex::add_2d(complex _other, complex& _output) {
	assert(size() == _other.size());
	for (size_t i = 0; i < size(); i++)
		_output[i] = at(i).add_1d(_other[i]);
}

complex complex::sub_2d(complex _other) {
	complex result = new_2d(size(), at(0).size());
	sub_2d(_other, result);
	return result;
}

void complex::sub_2d(complex _other, complex& _output) {
	assert(size() == _other.size());
	for (size_t i = 0; i < size(); i++)
		_output[i] = at(i).sub_1d(_other[i]);
}

complex complex::mul_2d(complex _other) {
	complex result = new_2d(size(), at(0).size());
	mul_2d(_other, result);
	return result;
}

void complex::mul_2d(complex _other, complex& _output) {
	assert(size() == _other.size());
	for (size_t i = 0; i < size(); i++)
		_output[i] = at(i).mul_1d(_other[i]);
}

complex complex::div_2d(complex _other) {
	complex result = new_2d(size(), at(0).size());
	div_2d(_other, result);
	return result;
}

void complex::div_2d(complex _other, complex& _output) {
	assert(size() == _other.size());
	for (size_t i = 0; i < size(); i++)
		_output[i] = at(i).div_1d(_other[i]);
}

complex complex::dot_2d(complex _other) {
	complex result = new_2d(height(), _other.width());
	dot_2d(_other, result);
	return result;
}

void complex::dot_2d(complex _other, complex& _output) {
	assert(width() == _other.height());
	for (size_t i = 0; i < height(); i++)
		for (size_t j = 0; j < _other.width(); j++)
			_output[i][j] = at(i).dot_1d(_other.col(j));
}

void complex::clone(complex& _output) {
	assert(size() == _output.size());
	_output.val.val() = val.val();
	for (size_t i = 0; i < size(); i++)
		at(i).clone(_output.at(i));
}

void complex::link(complex& _other) {
	val.link(_other.val);
	for (size_t i = 0; i < size(); i++)
		at(i).link(_other.at(i));
}

complex complex::clone() {
	complex result = val.val();
	result.resize(size());
	for (size_t i = 0; i < size(); i++)
		result.at(i) = at(i).clone();
	return result;
}

string complex::to_string() {
	if (size() == 0)
		return std::to_string(val.val());
	string result = "[";
	for (size_t i = 0; i < size() - 1; i++)
		result += at(i).to_string() + " ";
	result += at(size() - 1).to_string() + "]";
	return result;
}