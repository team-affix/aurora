#include "affix-base/pch.h"
#include "mse_loss.h"

using aurora::models::mse_loss;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::maths::tensor;

mse_loss::~mse_loss()
{

}

mse_loss::mse_loss()
{

}

mse_loss::mse_loss(
	const Model& a_model
) :
	m_model(a_model)
{

}

void mse_loss::param_recur(
	const function<void(Param&)>& a_func
)
{
	m_model->param_recur(a_func);
}

model* mse_loss::clone(
	const function<Param(Param&)>& a_func
)
{
	mse_loss* result = new mse_loss();
	result->m_model = m_model->clone(a_func);
	return result;
}

void mse_loss::fwd()
{
	m_model->fwd();
}

void mse_loss::bwd()
{
	m_model->bwd();
}

double mse_loss::signal(
	const tensor& a_y_des,
	const tensor* a_y,
	tensor* a_y_grad
)
{
	if (a_y == nullptr)
	{
		return signal(a_y_des, &m_y, &m_y_grad);
	}
	else if (a_y->size() != 0)
	{
		double l_result = 0;
		for (int i = 0; i < a_y->size(); i++)
			l_result += signal(a_y_des.at(i), &a_y->at(i), &a_y_grad->at(i));
		return l_result;
	}
	else
	{
		double l_result = pow(a_y_des.val() - a_y->val(), 2) / m_y_lowest_rank_count;
		a_y_grad->val() = m_2_over_y_lowest_rank_count * (a_y->val() - a_y_des.val());
		return l_result;
	}
	
}

double mse_loss::cycle(
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

void mse_loss::model_recur(
	const function<void(model*)>& a_func
)
{
	m_model->model_recur(a_func);
	a_func(this);
}

void mse_loss::compile()
{

	m_model->compile();

	m_x.group_link(m_model->m_x);
	m_x_grad.group_link(m_model->m_x_grad);
	m_y.group_link(m_model->m_y);
	m_y_grad.group_link(m_model->m_y_grad);

	m_y_lowest_rank_count = m_y.lowest_rank_count();
	m_2_over_y_lowest_rank_count = 2.0 / m_y_lowest_rank_count;
		
}
