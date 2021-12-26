#pragma once
#include "affix-base/pch.h"

#define MODEL_FIELDS \
virtual void param_recur(std::function<void(aurora::params::Param&)> a_func); \
virtual aurora::models::model* clone() { \
	return clone([](aurora::params::Param& pmt) { return pmt; }); \
} \
virtual model* clone(std::function<aurora::params::Param(aurora::params::Param&)> a_func); \
virtual void fwd(); \
virtual void bwd(); \
virtual void signal(const aurora::maths::tensor& a_y_des); \
virtual void model_recur(std::function<void(model*)> a_func); \
virtual void compile(); \
virtual aurora::maths::tensor& fwd(const aurora::maths::tensor& a_x) { \
	m_x.pop(a_x); \
	fwd(); \
	return m_y; \
} \
virtual aurora::maths::tensor& bwd(const aurora::maths::tensor& a_y_grad) { \
	m_y_grad.pop(a_y_grad); \
	bwd(); \
	return m_x_grad; \
} \
virtual void cycle(const aurora::maths::tensor& a_x, const aurora::maths::tensor& a_y_des) { \
	m_x.pop(a_x); \
	fwd(); \
	signal(a_y_des); \
	bwd(); \
} \

#define RECURRENT_FIELDS \
MODEL_FIELDS \
virtual void prep(size_t a_n); \
virtual void unroll(size_t a_n); \

#define ATTENTION_FIELDS \
MODEL_FIELDS \
virtual void prep(size_t a_a, size_t a_b); \
virtual void unroll(size_t a_a, size_t a_b); \
