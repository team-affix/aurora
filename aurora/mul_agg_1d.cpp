#include "affix-base/pch.h"
#include "mul_agg_1d.h"

using namespace aurora::params;
using namespace aurora::models;
using namespace aurora::maths;

mul_agg_1d::~mul_agg_1d() {

}

mul_agg_1d::mul_agg_1d(

)
{

}

mul_agg_1d::mul_agg_1d(
	const size_t& a_units
) :
	m_units(a_units)
{

}

void mul_agg_1d::param_recur(const std::function<void(Param&)>& a_func)
{

}

model* mul_agg_1d::clone(const std::function<Param(Param&)>& a_func)
{
	mul_agg_1d* result = new mul_agg_1d();
	result->m_units = m_units;
	return result;
}

void mul_agg_1d::fwd()
{
	m_y.val() = 1;

	for (int i = 0; i < m_x.size(); i++)
		m_y.val() *= m_x[i].val();

}

void mul_agg_1d::bwd()
{
	for (int i = 0; i < m_x_grad.size(); i++)
	{
		// To compute x_grad[i] given y_grad,
		// Imagine factoring such that y = x0(x1*x2*x3*x4*x5).
		// Now, it should be clear that dy/dx0 = (x1*x2*...)
		double l_every_other_x = 1;
		for (int j = 0; j < m_x.size(); j++)
		{
			if (j == i)
				continue;
			l_every_other_x *= m_x[j].val();
		}

		m_x_grad[i].val() = m_y_grad.val() * l_every_other_x;

	}

}

void mul_agg_1d::model_recur(const std::function<void(model*)>& a_func)
{
	a_func(this);
}

void mul_agg_1d::compile()
{
	m_x = tensor::new_1d(m_units);
	m_x_grad = tensor::new_1d(m_units);

}