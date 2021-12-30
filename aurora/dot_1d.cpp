#include "affix-base/pch.h"
#include "dot_1d.h"

using aurora::models::mul_1d;
using aurora::models::sum_1d;
using aurora::models::dot_1d;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::maths::tensor;

dot_1d::~dot_1d()
{

}

dot_1d::dot_1d(
	const size_t& a_units
)
{
	m_units = a_units;
	m_mul_1d = new mul_1d(m_units);
	m_sum_1d = new sum_1d(m_units);
}

void dot_1d::param_recur(
	const function<void(Param&)>& a_func
)
{
	m_mul_1d->param_recur(a_func);
	m_sum_1d->param_recur(a_func);
}

model* dot_1d::clone(
	const function<Param(Param&)>& a_func
)
{
	dot_1d* result = new dot_1d(m_units);
	return result;
}

void dot_1d::fwd()
{
	m_mul_1d->fwd();
	m_sum_1d->fwd();
}

void dot_1d::bwd()
{
	m_sum_1d->bwd();
	m_mul_1d->bwd();
}

void dot_1d::model_recur(
	const function<void(model*)>& a_func
)
{
	m_mul_1d->model_recur(a_func);
	m_sum_1d->model_recur(a_func);
	a_func(this);
}

void dot_1d::compile()
{
	m_x = tensor::new_2d(2, m_units);
	m_x_grad = tensor::new_2d(2, m_units);

	m_mul_1d->compile();
	m_sum_1d->compile();

	m_x.group_join_all_ranks(m_mul_1d->m_x);
	m_x_grad.group_join_all_ranks(m_mul_1d->m_x_grad);

	m_mul_1d->m_y.group_join_all_ranks(m_sum_1d->m_x);
	m_mul_1d->m_y_grad.group_join_all_ranks(m_sum_1d->m_x_grad);

	m_y.group_join_all_ranks(m_sum_1d->m_y);
	m_y_grad.group_join_all_ranks(m_sum_1d->m_y_grad);

}
