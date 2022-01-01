#include "add_1d.h"
#include "add_0d.h"

using aurora::models::add_1d;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::maths::tensor;

add_1d::~add_1d()
{

}

add_1d::add_1d()
{

}

add_1d::add_1d(
	const size_t& a_units
) :
	m_units(a_units)
{
	m_layer = new layer(a_units, new add_0d());
}

void add_1d::param_recur(
	const function<void(Param&)>& a_func
)
{
	m_layer->param_recur(a_func);
}

model* add_1d::clone(
	const function<Param(Param&)>& a_func
)
{
	add_1d* result = new add_1d();
	result->m_units = m_units;
	result->m_layer = m_layer->clone();
	return result;
}

void add_1d::fwd()
{
	m_layer->fwd();
}

void add_1d::bwd()
{
	m_layer->bwd();
}

void add_1d::model_recur(
	const function<void(model*)>& a_func
)
{
	m_layer->model_recur(a_func);
	a_func(this);
}

void add_1d::compile()
{

	m_x = tensor::new_2d(2, m_units);
	m_x_grad = tensor::new_2d(2, m_units);
	m_y = tensor::new_1d(m_units);
	m_y_grad = tensor::new_1d(m_units);

	m_layer->compile();

	for (int i = 0; i < m_units; i++)
	{
		m_x.col(i).link(m_layer->m_x[i]);
		m_x_grad.col(i).link(m_layer->m_x_grad[i]);
		m_y[i].link(m_layer->m_y[i]);
		m_y_grad[i].link(m_layer->m_y_grad[i]);
	}

}
