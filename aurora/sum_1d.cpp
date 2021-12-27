#include "affix-base/pch.h"
#include "sum_1d.h"

using aurora::models::sum_1d;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::maths::tensor;

sum_1d::~sum_1d() 
{

}

sum_1d::sum_1d(
	const size_t& a_units
)
{
	m_units = a_units;
}

void sum_1d::param_recur(
	const function<void(Param&)>& a_func
)
{

}

model* sum_1d::clone(
	const function<Param(Param&)>& a_func
)
{
	sum_1d* result = new sum_1d(m_units);
	return result;
}

void sum_1d::fwd() 
{
	m_y.val() = m_x.sum_1d();
}

void sum_1d::bwd() 
{

}

void sum_1d::signal(
	const tensor& a_y_des
) 
{
	m_y_grad.val() = m_y.val() - a_y_des.val();
}

void sum_1d::model_recur(
	const function<void(model*)>& a_func
)
{
	a_func(this);
}

void sum_1d::compile() 
{
	m_x = tensor::new_1d(m_units);
	m_x_grad = tensor::new_1d(m_units);

	for (int i = 0; i < m_units; i++)
		m_x_grad[i].group_join_all_ranks(m_y_grad);

}