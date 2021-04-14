#pragma once

#define MODEL_FIELDS \
virtual void pmt_wise(function<void(ptr<param>&)> a_func); \
virtual model* clone(); \
virtual model* clone(function<void(ptr<param>&)> a_init); \
virtual void fwd(); \
virtual void bwd(); \
virtual tensor& fwd(tensor& a_x); \
virtual tensor& bwd(tensor& a_y_grad); \
virtual void signal(tensor& a_y_des); \
virtual void cycle(tensor& a_x, tensor& a_y_des); \
virtual void recur(function<void(model*)> a_func); \
virtual void compile(); \

#define RECURRENT_FIELDS \
MODEL_FIELDS \
virtual void prep(size_t a_n); \
virtual void unroll(size_t a_n); \

#define ATTENTION_FIELDS \
MODEL_FIELDS \
virtual void prep(size_t a_a, size_t a_b); \
virtual void unroll(size_t a_a, size_t a_b); \
