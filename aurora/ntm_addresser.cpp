#include "pch.h"
#include "ntm_addresser.h"

using aurora::models::ntm_addresser;

ntm_addresser::~ntm_addresser() {

}

ntm_addresser::ntm_addresser() {

}

ntm_addresser::ntm_addresser(size_t a_height, size_t a_width, vector<int> a_valid_shifts) {
	m_height = a_height;
	m_width = a_width;
	valid_shifts = a_valid_shifts;
}

void ntm_addresser::pmt_wise(function<void(ptr<param>&)> a_func) {

}

model* ntm_addresser::clone() {
	ntm_addresser* result = new ntm_addresser(m_height, m_width, valid_shifts);
	return result;
}

model* ntm_addresser::clone(function<void(ptr<param>&)> a_func) {
	ntm_addresser* result = new ntm_addresser(m_height, m_width, valid_shifts);
	return result;
}

void ntm_addresser::fwd() {
	fwd_similar();
	fwd_sparse();
	fwd_sparse_normalize();
	fwd_interpolate();
	fwd_shift();
	fwd_sharp();
	fwd_sharp_normalize();
}

void ntm_addresser::fwd_similar() {
	key_magnitude = k.mag_1d();
	for (int i = 0; i < m.height(); i++) {
		memory_magnitude_vector[i].val() = m[i].mag_1d();
		magnitude_product_vector[i].val() = key_magnitude * memory_magnitude_vector[i];
		dot_product_vector[i].val() = k.dot_1d(m[i]);
		similar[i].val() = dot_product_vector[i] / (key_magnitude * memory_magnitude_vector[i]);
	}
}

void ntm_addresser::fwd_sparse() {
	for (int i = 0; i < similar.size(); i++)
		sparse[i].val() = exp(beta[0] * similar[i]);
}

void ntm_addresser::fwd_sparse_normalize() {
	sparse_sum = sparse.sum_1d();
	for (int i = 0; i < sparse.size(); i++)
		sparse_normalize[i].val() = sparse[i] / sparse_sum;
}

void ntm_addresser::fwd_interpolate() {
	for (int i = 0; i < interpolate.size(); i++)
		interpolate[i].val() = g[0] * sparse_normalize[i] + (1 - g[0]) * wx[i];
}

void ntm_addresser::fwd_shift() {
	shift.clear();
	for (int i = 0; i < shift.size(); i++)
		for (int j = 0; j < valid_shifts.size(); j++) {
			int shift_amount = valid_shifts[j];
			int src = i;
			int dst = positive_modulo((i + shift_amount), (int)shift.size());
			double& src_val = interpolate[src];
			double& dst_val = shift[dst];
			dst_val += src_val * s[j];
		}
}

void ntm_addresser::fwd_sharp() {
	for (int i = 0; i < sharpen.size(); i++)
		sharpen[i].val() = pow(shift[i], gamma[0]);
}

void ntm_addresser::fwd_sharp_normalize() {
	sharpen_sum = sharpen.sum_1d();
	for (int i = 0; i < sparse.size(); i++)
		sharp_normalize[i].val() = sharpen[i] / sharpen_sum;
}

void ntm_addresser::bwd() {
	sharp_normalize_grad.add_1d(y_grad, sharp_normalize_grad);
	sharp_normalize_grad.add_1d(wy_grad, sharp_normalize_grad);
	bwd_sharp_normalize();
	sharp_normalize_grad.clear();
	bwd_sharp();
	bwd_shift();
	bwd_interpolate();
	bwd_sparse_normalize();
	bwd_sparse();
	bwd_similar();
}


void ntm_addresser::bwd_similar() {
	k_grad.clear();
	m_grad.clear();
	for (int i = 0; i < k.size(); i++)
		for (int j = 0; j < m.height(); j++) {
			tensor& slot = m[j];
			double& mag_product = magnitude_product_vector[j];
			double& dot_product = dot_product_vector[j];
			double& mem_mag = memory_magnitude_vector[j];
			double k_a = slot[i] / mag_product;
			double k_b = dot_product * k[i];
			double k_c = mem_mag * pow(key_magnitude, 3);
			k_grad[i].val() += similar_grad[j] * (k_a - k_b / k_c);
			double m_a = k[i] / mag_product;
			double m_b = dot_product * slot[i];
			double m_c = key_magnitude * pow(mem_mag, 3);
			m_grad[j][i].val() += similar_grad[j] * (m_a - m_b / m_c);
		}
}

void ntm_addresser::bwd_sparse() {
	beta_grad.clear();
	for (int i = 0; i < sparse_grad.size(); i++) {
		similar_grad[i].val() = beta[0] * sparse_grad[i];
		beta_grad[0].val() += similar[i] * sparse_grad[i];
	}
}

void ntm_addresser::bwd_sparse_normalize() {
	for (int i = 0; i < sparse_normalize_grad.size(); i++)
		sparse_grad[i].val() =
			sparse_normalize_grad[i] * (sparse_sum - sparse[i]) / pow(sparse_sum, 2);
}

void ntm_addresser::bwd_interpolate() {
	g_grad.clear();
	for (int i = 0; i < interpolate_grad.size(); i++) {
		sparse_normalize_grad[i].val() = interpolate_grad[i] * g[0];
		wx_grad[i].val() = interpolate_grad[i] * (1 - g[0]);
		g_grad[0] += interpolate_grad[i] * (sparse_normalize[i] - wx[i]);
	}
}

void ntm_addresser::bwd_shift() {
	interpolate_grad.clear();
	s_grad.clear();
	for (int i = 0; i < shift_grad.size(); i++)
		for (int j = 0; j < valid_shifts.size(); j++) {
			int shift_amount = valid_shifts[j];
			int src = i;
			int dst = positive_modulo((i - shift_amount), (int)shift.size());
			double& src_val = shift_grad[src];
			double& dst_val = interpolate_grad[dst];
			dst_val += src_val * s[j];
			s_grad[j] += src_val * interpolate[dst];
		}
}

void ntm_addresser::bwd_sharp() {
	gamma_grad.clear();
	for (int i = 0; i < sharpen_grad.size(); i++) {
		shift_grad[i].val() = sharpen_grad[i] * gamma[0] * pow(shift[i], gamma[0] - 1);
		gamma_grad[0].val() += sharpen_grad[i] * log(shift[i]) * sharpen[i];
	}
}

void ntm_addresser::bwd_sharp_normalize() {
	for (int i = 0; i < sharp_normalize_grad.size(); i++)
		sharpen_grad[i].val() =
		sharp_normalize_grad[i] * (sharpen_sum - sharpen[i]) / pow(sharpen_sum, 2);
}

tensor& ntm_addresser::fwd(tensor& a_x) {
	x.pop(a_x);
	fwd();
	return y;
}

tensor& ntm_addresser::bwd(tensor& a_y_grad) {
	y_grad.pop(a_y_grad);
	bwd();
	return x_grad;
}

void ntm_addresser::signal(tensor& a_y_des) {
	y.sub_1d(a_y_des, y_grad);
}

void ntm_addresser::cycle(tensor& a_x, tensor& a_y_des) {
	x.pop(a_x);
	fwd();
	signal(a_y_des);
	bwd();
}

void ntm_addresser::recur(function<void(model*)> a_func) {

}

void ntm_addresser::compile() {

	y = tensor::new_1d(m_height);
	y_grad = tensor::new_1d(m_height);

	m = tensor::new_2d(m_height, m_width);
	m_grad = tensor::new_2d(m_height, m_width);
	k = tensor::new_1d(m_width);
	k_grad = tensor::new_1d(m_width);
	beta = tensor::new_1d(1);
	beta_grad = tensor::new_1d(1);
	g = tensor::new_1d(1);
	g_grad = tensor::new_1d(1);
	s = tensor::new_1d(valid_shifts.size());
	s_grad = tensor::new_1d(valid_shifts.size());
	gamma = tensor::new_1d(1);
	gamma_grad = tensor::new_1d(1);

	wx = tensor::new_1d(m_height);
	wx_grad = tensor::new_1d(m_height);
	wy = tensor::new_1d(m_height);
	wy_grad = tensor::new_1d(m_height);

	memory_magnitude_vector = tensor::new_1d(m_height);
	magnitude_product_vector = tensor::new_1d(m_height);
	dot_product_vector = tensor::new_1d(m_height);
	similar = tensor::new_1d(m_height);
	similar_grad = tensor::new_1d(m_height);

	sparse = tensor::new_1d(m_height);
	sparse_grad = tensor::new_1d(m_height);

	sparse_normalize = tensor::new_1d(m_height);
	sparse_normalize_grad = tensor::new_1d(m_height);

	interpolate = tensor::new_1d(m_height);
	interpolate_grad = tensor::new_1d(m_height);

	shift = tensor::new_1d(m_height);
	shift_grad = tensor::new_1d(m_height);

	sharpen = tensor::new_1d(m_height);
	sharpen_grad = tensor::new_1d(m_height);

	sharp_normalize = tensor::new_1d(m_height);
	sharp_normalize_grad = tensor::new_1d(m_height);

	y.group_join(wy);
	y.group_join(sharp_normalize);

}

int ntm_addresser::positive_modulo(int i, int n) {
	return (i % n + n) % n;
}