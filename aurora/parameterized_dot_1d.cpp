#include "affix-base/pch.h"
#include "parameterized_dot_1d.h"
#include "weight.h"

using aurora::models::parameterized_dot_1d;
using std::function;
using aurora::maths::tensor;
using aurora::params::Param;
using aurora::models::model;

parameterized_dot_1d::~parameterized_dot_1d()
{

}

parameterized_dot_1d::parameterized_dot_1d(
	const size_t& a_units
)
{
	m_units = a_units;
	m_layer = new layer(m_units, new weight());
	m_sum_1d = new sum_1d(m_units);
}

void parameterized_dot_1d::param_recur(const function<void(Param&)>& a_func)
{
	m_layer->param_recur(a_func);
	m_sum_1d->param_recur(a_func);
}

model* parameterized_dot_1d::clone(const function<Param(Param&)>& a_func)
{
	parameterized_dot_1d* result = new parameterized_dot_1d(m_units);
	result->m_layer = m_layer->clone(a_func);
	return result;
}

void parameterized_dot_1d::fwd()
{
	m_layer->fwd();
	m_sum_1d->fwd();
}

void parameterized_dot_1d::bwd()
{
	m_sum_1d->bwd();
	m_layer->bwd();
}

void parameterized_dot_1d::signal(const tensor& a_y_des)
{
	m_y_grad.val() = m_y.val() - a_y_des.val();
}

void parameterized_dot_1d::model_recur(const function<void(model*)>& a_func)
{
	m_layer->model_recur(a_func);
	m_sum_1d->model_recur(a_func);
	a_func(this);
}

void parameterized_dot_1d::compile()
{

	m_x = tensor::new_1d(m_units);
	m_x_grad = tensor::new_1d(m_units);

	m_layer->compile();
	m_sum_1d->compile();

	m_x.group_join_all_ranks(m_layer->m_x);
	m_x_grad.group_join_all_ranks(m_layer->m_x_grad);

	m_layer->m_y.group_join_all_ranks(m_sum_1d->m_x);
	m_layer->m_y_grad.group_join_all_ranks(m_sum_1d->m_x_grad);

	m_y.group_join_all_ranks(m_sum_1d->m_y);
	m_y_grad.group_join_all_ranks(m_sum_1d->m_y_grad);

}
