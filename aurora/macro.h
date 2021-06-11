#pragma once
#include "pch.h"

#define MODEL_FIELDS \
virtual void param_recur(function<void(ptr<param>&)> a_func); \
virtual model* clone(); \
virtual model* clone(function<void(ptr<param>&)> a_func); \
virtual void fwd(); \
virtual void bwd(); \
virtual void signal(tensor& a_y_des); \
virtual void model_recur(function<void(model*)> a_func); \
virtual void compile(); \
virtual tensor& fwd(tensor& a_x) { \
	x.pop(a_x); \
	fwd(); \
	return y; \
} \
virtual tensor& bwd(tensor& a_y_grad) { \
	y_grad.pop(a_y_grad); \
	bwd(); \
	return x_grad; \
} \
virtual void cycle(tensor& a_x, tensor& a_y_des) { \
	x.pop(a_x); \
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
