#include "add_0d.h"

using aurora::models::add_0d;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::maths::tensor;

add_0d::~add_0d()
{

}

add_0d::add_0d()
{

}

void add_0d::param_recur(
	const function<void(Param&)>& a_func
)
{

}

model* add_0d::clone(
	const function<Param(Param&)>& a_func
)
{
	add_0d* result = new add_0d();
	return result;
}

void add_0d::fwd()
{
	m_y.val() = m_x[0].val() + m_x[1].val();
}

void add_0d::bwd()
{

}

void add_0d::model_recur(
	const function<void(model*)>& a_func
)
{
	a_func(this);
}

void add_0d::compile()
{

	m_x = tensor::new_1d(2);
	m_x_grad = tensor::new_1d(2);
	
	m_x_grad[0].group_join_all_ranks(m_y_grad);
	m_x_grad[1].group_join_all_ranks(m_y_grad);

}
