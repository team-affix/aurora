#include "affix-base/pch.h"
#include "mul_0d.h"

using aurora::models::mul_0d;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::maths::tensor;

mul_0d::~mul_0d()
{

}

mul_0d::mul_0d()
{

}

void mul_0d::param_recur(
	const function<void(Param&)>& a_func
)
{

}

model* mul_0d::clone(
	const function<Param(Param&)>& a_func
)
{
	mul_0d* result = new mul_0d();
	return result;
}

void mul_0d::fwd()
{
	m_y.val() = m_x[0].val() * m_x[1].val();
}

void mul_0d::bwd()
{
	m_x_grad[0].val() = m_y_grad.val() * m_x[1].val();
	m_x_grad[1].val() = m_y_grad.val() * m_x[0].val();
}

void mul_0d::model_recur(
	const function<void(model*)>& a_func
)
{
	a_func(this);
}

void mul_0d::compile()
{
	m_x = tensor::new_1d(2);
	m_x_grad = tensor::new_1d(2);
}
