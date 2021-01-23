#pragma once

#define MODEL_FIELDS \
virtual model* clone(); \
virtual model* clone(vector<param*>& a_pl); \
virtual model* clone(vector<param_sgd*>& a_pl); \
virtual model* clone(vector<param_mom*>& a_pl); \
virtual void fwd(); \
virtual void bwd(); \
virtual tensor& fwd(tensor a_x); \
virtual tensor& bwd(tensor a_y_grad); \
virtual void signal(tensor a_y_des); \
virtual void cycle(tensor a_x, tensor a_y_des); \
virtual void recur(function<void(model*)> a_func); \
virtual void compile(); \

#define RECURRENT_FIELDS \
MODEL_FIELDS \
virtual void prep(size_t a_n); \
virtual void unroll(size_t a_n); \