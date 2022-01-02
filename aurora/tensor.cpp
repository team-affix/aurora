#include "affix-base/pch.h"
#include "tensor.h"

using namespace aurora;
using namespace maths;

double& tensor::val() {
	return m_val_ptr.val();
}

double tensor::val() const {
	return m_val_ptr.val();
}

std::vector<tensor>& tensor::vec() {
	return m_vec_ptr.val();
}

const std::vector<tensor>& tensor::vec() const {
	return m_vec_ptr.val();
}

tensor::~tensor() {

}

tensor::tensor() {

}

tensor::tensor(const double& a_val) {
	this->val() = a_val;
}

tensor::tensor(const std::vector<tensor>& a_vec) {
	std::copy(a_vec.begin(), a_vec.end(), back_inserter(vec()));
}

tensor::tensor(const std::initializer_list<tensor>& a_il) {
	std::copy(a_il.begin(), a_il.end(), back_inserter(vec()));
}

void tensor::set(const tensor& a_other) {
	val() = a_other.val();
	resize(a_other.size());
	for (size_t i = 0; i < vec().size(); i++)
		at(i).set(a_other.at(i));
}

void tensor::pop(const tensor& a_other) {
	val() = a_other.val();
	for (size_t i = 0; i < a_other.size(); i++)
		at(i).pop(a_other.at(i));
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

tensor tensor::new_1d(size_t a_a, std::uniform_real_distribution<double>& a_urd, std::default_random_engine& a_re) {
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

tensor tensor::new_2d(size_t a_a, size_t a_b, std::uniform_real_distribution<double>& a_urd, std::default_random_engine& a_re) {
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

void tensor::sum_1d(double& a_output) {
	for (int i = 0; i < vec().size(); i++)
		a_output += vec().at(i).val();
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

void tensor::norm_1d(tensor& a_output) {

	a_output = clone();

	double l_min = min_1d();
	if (l_min < 0)
		// OFFSET BY NEGATIVE VALUES FIRST
		a_output.add_1d(tensor::new_1d(a_output.size(), l_min), a_output);

	double l_sum = a_output.sum_1d();
	for (int i = 0; i < size(); i++) {
		a_output[i].val() = a_output[i].val() / l_sum;
	}

}

void tensor::signed_norm_1d(tensor& a_output)
{
	double l_sum = abs_1d().sum_1d();
	for (int i = 0; i < a_output.size(); i++)
		a_output[i].val() = at(i).val() / l_sum;
}

void tensor::norm_2d(tensor& a_output) {

	a_output = clone();

	double l_min = min_2d();
	if (l_min < 0)
		// OFFSET BY NEGATIVE VALUES FIRST
		a_output.add_2d(tensor::new_2d(a_output.height(), a_output.width(), l_min), a_output);

	double l_sum = a_output.sum_2d();
	for (int i = 0; i < height(); i++) {
		for (int j = 0; j < width(); j++) {
			a_output[i][j].val() = a_output[i][j].val() / l_sum;
		}
	}

}

double tensor::mag_1d() {
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

double tensor::sum_1d() {
	double result = 0;
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
			result[i * w + j].group_link(at(i)[j]);
	return result;
}

tensor tensor::roll(size_t a_width) {
	size_t s = vec().size();
	assert(s % a_width == 0);
	size_t h = s / a_width;
	tensor result = new_2d(h, a_width);
	for (int i = 0; i < vec().size(); i++)
		result[i / a_width][i % a_width].group_link(at(i));
	return result;
}

size_t tensor::width() const {
	return vec().at(0).size();
}

size_t tensor::height() const {
	return size();
}

double tensor::max_1d() {
	double result = -INFINITY;
	for (int i = 0; i < size(); i++)
		if (at(i) > result)
			result = at(i);
	return result;
}

double tensor::min_1d() {
	double result = INFINITY;
	for (int i = 0; i < size(); i++)
		if (at(i) < result)
			result = at(i);
	return result;
}

size_t tensor::arg_max_1d() {
	double val = -INFINITY;
	size_t result = -1;
	for (int i = 0; i < size(); i++)
		if (at(i) > val) {
			val = at(i);
			result = i;
		}
	return result;
}

size_t tensor::arg_min_1d() {
	double val = INFINITY;
	size_t result = -1;
	for (int i = 0; i < size(); i++)
		if (at(i) < val) {
			val = at(i);
			result = i;
		}
	return result;
}

double tensor::max_2d()
{
	double result = -INFINITY;
	for (int i = 0; i < size(); i++)
	{
		double l_max_1d = at(i).max_1d();
		if (l_max_1d > result)
			result = l_max_1d;
	}
	return result;
}

double tensor::min_2d()
{
	double result = INFINITY;
	for (int i = 0; i < size(); i++)
	{
		double l_min_1d = at(i).min_1d();
		if (l_min_1d < result)
			result = l_min_1d;
	}
	return result;
}

tensor tensor::norm_1d() {
	tensor result = tensor::new_1d(size());
	norm_1d(result);
	return result;
}

tensor tensor::signed_norm_1d()
{
	tensor result = tensor::new_1d(size());
	signed_norm_1d(result);
	return result;
}

tensor tensor::norm_2d() {
	tensor result = tensor::new_2d(height(), width());
	norm_2d(result);
	return result;
}

tensor tensor::row(size_t a_a) {
	tensor result;
	result.group_link(at(a_a));
	return result;
}

tensor tensor::col(size_t a_a) {
	tensor result = new_1d(vec().size());
	for (int i = 0; i < size(); i++)
		result[i].group_link(at(i)[a_a]);
	return result;
}

tensor tensor::range(size_t a_start, size_t a_len) {
	tensor result = new_1d(a_len);
	for (int i = 0; i < a_len; i++)
	{
		size_t src = a_start + i;
		size_t dst = i;
		result[dst].group_link(at(src));
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
		result[dst].group_link(row_section);
	}
	return result;
}

tensor tensor::clone_row(size_t a_a) const {
	return at(a_a).clone();
}

tensor tensor::clone_col(size_t a_a) const {
	tensor result = new_1d(vec().size());
	for (int i = 0; i < size(); i++)
		result[i].set(at(i).at(a_a).clone());
	return result;
}

tensor tensor::clone_range(size_t a_start, size_t a_len) const {
	tensor result = new_1d(a_len);
	for (int i = 0; i < a_len; i++)
	{
		size_t src = a_start + i;
		size_t dst = i;
		result[dst] = vec().at(src);
	}
	return result.clone();
}

tensor tensor::add_1d(const tensor& a_other) {
	tensor result = new_1d(vec().size());
	add_1d(a_other, result);
	return result;
}

void tensor::add_1d(const tensor& a_other, tensor& a_output) {
	assert(vec().size() == a_other.size());
	for (int i = 0; i < vec().size(); i++)
		a_output[i].val() = at(i).val() + a_other.at(i).val();
}

tensor tensor::sub_1d(const tensor& a_other) {
	tensor result = new_1d(vec().size());
	sub_1d(a_other, result);
	return result;
}

void tensor::sub_1d(const tensor& a_other, tensor& a_output) {
	assert(vec().size() == a_other.vec().size());
	for (int i = 0; i < vec().size(); i++)
		a_output[i].val() = vec().at(i).val() - a_other.vec().at(i).val();
}

tensor tensor::mul_1d(const tensor& a_other) {
	tensor result = new_1d(vec().size());
	mul_1d(a_other, result);
	return result;
}

void tensor::mul_1d(const tensor& a_other, tensor& a_output) {
	assert(vec().size() == a_other.vec().size());
	for (int i = 0; i < vec().size(); i++)
		a_output[i].val() = vec().at(i).val() * a_other.vec().at(i).val();
}

tensor tensor::div_1d(const tensor& a_other) {
	tensor result = new_1d(vec().size());
	div_1d(a_other, result);
	return result;
}

tensor tensor::pow_1d(const tensor& a_other) {
	tensor result = new_1d(vec().size());
	pow_1d(a_other, result);
	return result;
}

void tensor::div_1d(const tensor& a_other, tensor& a_output) {
	assert(vec().size() == a_other.vec().size());
	for (int i = 0; i < vec().size(); i++)
		a_output[i].val() = vec().at(i).val() / a_other.vec().at(i).val();
}

void tensor::pow_1d(const tensor& a_other, tensor& a_output) {
	assert(vec().size() == a_other.vec().size());
	for (int i = 0; i < vec().size(); i++)
		a_output[i].val() = pow(at(i).val(), a_other.at(i).val());
}

double tensor::dot_1d(const tensor& a_other) {
	double result;
	dot_1d(a_other, result);
	return result;
}

void tensor::dot_1d(const tensor& a_other, double& a_output) {
	assert(vec().size() == a_other.vec().size());
	a_output = (double)0;
	for (int i = 0; i < vec().size(); i++)
		a_output += vec().at(i).val() * a_other.vec().at(i).val();
}

tensor tensor::add_2d(const tensor& a_other) {
	tensor result = new_2d(vec().size(), vec().at(0).vec().size());
	add_2d(a_other, result);
	return result;
}

void tensor::add_2d(const tensor& a_other, tensor& a_output) {
	assert(vec().size() == a_other.vec().size());
	for (int i = 0; i < vec().size(); i++)
		at(i).add_1d(a_other.at(i), a_output.at(i));
}

tensor tensor::sub_2d(const tensor& a_other) {
	tensor result = new_2d(vec().size(), vec().at(0).size());
	sub_2d(a_other, result);
	return result;
}

void tensor::sub_2d(const tensor& a_other, tensor& a_output) {
	assert(vec().size() == a_other.vec().size());
	for (int i = 0; i < vec().size(); i++)
		at(i).sub_1d(a_other.at(i), a_output.at(i));
}

tensor tensor::mul_2d(const tensor& a_other) {
	tensor result = new_2d(vec().size(), vec().at(0).size());
	mul_2d(a_other, result);
	return result;
}

void tensor::mul_2d(const tensor& a_other, tensor& a_output) {
	assert(vec().size() == a_other.vec().size());
	for (int i = 0; i < vec().size(); i++)
		at(i).mul_1d(a_other.at(i), a_output.at(i));
}

tensor tensor::div_2d(const tensor& a_other) {
	tensor result = new_2d(vec().size(), vec().at(0).size());
	div_2d(a_other, result);
	return result;
}

tensor tensor::pow_2d(const tensor& a_other) {
	tensor result = new_2d(vec().size(), vec().at(0).size());
	pow_2d(a_other, result);
	return result;
}

void tensor::div_2d(const tensor& a_other, tensor& a_output) {
	assert(vec().size() == a_other.vec().size());
	for (int i = 0; i < vec().size(); i++)
		at(i).div_1d(a_other.at(i), a_output.at(i));
}

void tensor::pow_2d(const tensor& a_other, tensor& a_output) {
	assert(vec().size() == a_other.vec().size());
	for (int i = 0; i < vec().size(); i++)
		at(i).pow_1d(a_other.at(i), a_output.at(i));
}

tensor tensor::dot_2d(const tensor& a_other) {
	tensor result = new_2d(height(), a_other.width());
	dot_2d(a_other, result);
	return result;
}

void tensor::dot_2d(const tensor& a_other, tensor& a_output) {
	assert(width() == a_other.height());
	for (int i = 0; i < height(); i++)
		for (size_t j = 0; j < a_other.width(); j++)
		row(i).dot_1d(a_other.clone_col(j), a_output[i][j]);
}

void tensor::cat(tensor& a_other, tensor& a_output) {
	for (int i = 0; i < size(); i++)
		a_output[i].group_link(at(i));
	for (int i = 0; i < a_other.size(); i++)
		a_output[i + size()].group_link(a_other.at(i));
}

double tensor::cos_sim(tensor& a_other) {
	double dot = abs(dot_1d(a_other));
	double mag_a = mag_1d();
	double mag_b = a_other.mag_1d();
	return dot / (mag_a * mag_b);
}

tensor tensor::cat(tensor& a_other) {
	tensor result = tensor::new_1d(size() + a_other.size());
	cat(a_other, result);
	return result;
}

void tensor::rank_recur(const std::function<void(tensor*)>& a_func) {
	if (size() != 0)
		for (int i = 0; i < size(); i++)
			at(i).rank_recur(a_func);
	a_func(this);
}

void tensor::lowest_rank_recur(
	const std::function<void(tensor*)>& a_func
)
{
	if (size() != 0)
		for (int i = 0; i < size(); i++)
			at(i).lowest_rank_recur(a_func);
	else
		a_func(this);
}

size_t tensor::lowest_rank_count()
{
	size_t l_result = 0;
	lowest_rank_recur(
		[&](tensor*)
		{
			l_result++;
		});
	return l_result;
}

void tensor::link(
	tensor& a_other
)
{
	m_val_ptr.link(a_other.m_val_ptr);

	resize(a_other.size());

	for (int i = 0; i < size(); i++)
		at(i).link(a_other.at(i));

}

void tensor::unlink()
{
	m_val_ptr.unlink();
	m_vec_ptr.unlink();
}

void tensor::group_link(
	tensor& a_other
)
{

	m_val_ptr.group_link(a_other.m_val_ptr);

	resize(a_other.size());

	for (int i = 0; i < size(); i++)
		at(i).group_link(a_other.at(i));

}

void tensor::group_unlink()
{
	m_val_ptr.group_unlink();
	m_vec_ptr.group_unlink();
}

tensor tensor::clone() const {
	tensor result = tensor(val());
	result.vec().resize(vec().size());
	for (int i = 0; i < vec().size(); i++)
		result.at(i) = at(i).clone();
	return result;
}

tensor tensor::group_link() {
	tensor result = tensor();
	result.group_link(*this);
	return result;
}

std::string tensor::to_string() const
{
	if (vec().size() == 0)
		return std::to_string(val());
	std::string result = "[";
	for (int i = 0; i < vec().size() - 1; i++)
		result += vec().at(i).to_string() + " ";
	result += vec().at(vec().size() - 1).to_string() + "]";
	return result;
}

void tensor::clear() {
	val() = 0.0;
	for (int i = 0; i < vec().size(); i++)
		at(i).clear();
}

size_t tensor::size() const {
	return vec().size();
}

tensor& tensor::at(size_t a_a) {
	return vec().at(a_a);
}

const tensor& tensor::at(size_t a_a) const {
	return vec().at(a_a);
}

tensor::operator double& () {
	return val();
}

tensor::operator const double& () const {
	return val();
}

tensor& tensor::operator[](size_t a) {
	return vec()[a];
}

const tensor& tensor::operator[](size_t a) const {
	return at(a);
}
