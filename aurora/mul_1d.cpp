#include "affix-base/pch.h"
#include "mul_1d.h"
#include "mul_0d.h"

using aurora::models::mul_1d;
using aurora::models::mul_0d;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::maths::tensor;

mul_1d::~mul_1d()
{

}

mul_1d::mul_1d(
	const size_t& a_units
)
{
	m_units = a_units;
	m_layer = new layer(m_units, new mul_0d());
}

void mul_1d::param_recur(
	const function<void(Param&)>& a_func
)
{
	m_layer->param_recur(a_func);
}

model* mul_1d::clone(
	const function<Param(Param&)>& a_func
)
{
	mul_1d* result = new mul_1d(m_units);
	return result;
}

void mul_1d::fwd()
{
	m_layer->fwd();
}

void mul_1d::bwd()
{
	m_layer->bwd();
}

void mul_1d::model_recur(
	const function<void(model*)>& a_func
)
{
	m_layer->model_recur(a_func);
	a_func(this);
}

void mul_1d::compile()
{
	m_x = tensor::new_2d(2, m_units);
	m_x_grad = tensor::new_2d(2, m_units);
	m_y = tensor::new_1d(m_units);
	m_y_grad = tensor::new_1d(m_units);

	m_layer->compile();

	for (int i = 0; i < m_units; i++)
	{
		m_x[0][i].group_join_all_ranks(m_layer->m_x[i][0]);
		m_x_grad[0][i].group_join_all_ranks(m_layer->m_x_grad[i][0]);
		m_x[1][i].group_join_all_ranks(m_layer->m_x[i][1]);
		m_x_grad[1][i].group_join_all_ranks(m_layer->m_x_grad[i][1]);
	}

	m_y.group_join_all_ranks(m_layer->m_y);
	m_y_grad.group_join_all_ranks(m_layer->m_y_grad);

}