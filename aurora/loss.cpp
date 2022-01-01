#include "loss.h"

using aurora::models::loss;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::maths::tensor;

loss::~loss()
{

}

loss::loss()
{

}

loss::loss(
	const Model& a_model
) :
	m_model(a_model)
{

}

void loss::param_recur(
	const function<void(Param&)>& a_func
)
{
	m_model->param_recur(a_func);
}

model* loss::clone(
	const function<Param(Param&)>& a_func
)
{
	loss* result = new loss();
	result->m_model = m_model->clone(a_func);
	return result;
}

void loss::fwd()
{
	m_model->fwd();
}

void loss::bwd()
{
	m_model->bwd();
}

double loss::signal(
	const tensor& a_y_des,
	const tensor* a_y,
	tensor* a_y_grad
)
{
	return 0;
}

double loss::cycle(
	const tensor& a_x,
	const tensor& a_y_des
)
{
	m_x.pop(a_x);
	fwd();
	double l_result = signal(a_y_des);
	bwd();
	return l_result;
}

void loss::model_recur(
	const function<void(model*)>& a_func
)
{
	m_model->model_recur(a_func);
	a_func(this);
}

void loss::compile()
{

	m_model->compile();

	m_x.link(m_model->m_x);
	m_x_grad.link(m_model->m_x_grad);
	m_y.link(m_model->m_y);
	m_y_grad.link(m_model->m_y_grad);

}
