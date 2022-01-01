#include "affix-base/pch.h"
#include "rounding_spc.h"
#include "static_vals.h"

using aurora::models::rounding_spc;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::maths::tensor;

std::uniform_real_distribution<double> rounding_spc::s_urd(0, 1);

rounding_spc::~rounding_spc()
{

}

rounding_spc::rounding_spc()
{

}

void rounding_spc::param_recur(
	const function<void(Param&)>& a_func
)
{

}

model* rounding_spc::clone(
	const function<Param(Param&)>& a_func
)
{
	rounding_spc* result = new rounding_spc();
	return result;
}

void rounding_spc::fwd()
{
	double l_lower = std::floor(m_x.val());
	double l_decimal = m_x.val() - l_lower;
	double l_upper = l_lower + 1.0;

	if (s_urd(static_vals::random_engine) <= l_decimal)
	{
		m_y.val() = l_upper;
	}
	else
	{
		m_y.val() = l_lower;
	}

}

void rounding_spc::bwd()
{

}

void rounding_spc::model_recur(
	const function<void(model*)>& a_func
)
{
	a_func(this);
}

void rounding_spc::compile()
{

	m_x_grad.link(m_y_grad);

}