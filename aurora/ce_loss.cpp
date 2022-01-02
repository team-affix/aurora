#include "affix-base/pch.h"
#include "ce_loss.h"

using aurora::models::ce_loss;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::maths::tensor;

ce_loss::~ce_loss()
{

}

ce_loss::ce_loss()
{

}

ce_loss::ce_loss(
	const Model& a_model
) :
	m_model(a_model)
{

}

void ce_loss::param_recur(
	const function<void(Param&)>& a_func
)
{
	m_model->param_recur(a_func);
}

model* ce_loss::clone(
	const function<Param(Param&)>& a_func
)
{
	ce_loss* result = new ce_loss();
	result->m_model = m_model->clone(a_func);
	return result;
}

void ce_loss::fwd()
{
	m_model->fwd();
}

void ce_loss::bwd()
{
	m_model->bwd();
}

double ce_loss::signal(
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
		double l_desired_compliment = 1.0 - a_y_des.val();
		double l_predicted_compliment = 1.0 - a_y->val();
		
		double l_result = -(a_y_des.val() * log(a_y->val()) + l_desired_compliment * log(l_predicted_compliment));
		a_y_grad->val() = -(a_y_des.val() / a_y->val() - l_desired_compliment / l_predicted_compliment);

		return l_result;
	}

}

double ce_loss::cycle(
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

void ce_loss::model_recur(
	const function<void(model*)>& a_func
)
{
	m_model->model_recur(a_func);
	a_func(this);
}

void ce_loss::compile()
{

	m_model->compile();

	m_x.group_link(m_model->m_x);
	m_x_grad.group_link(m_model->m_x_grad);
	m_y.group_link(m_model->m_y);
	m_y_grad.group_link(m_model->m_y_grad);

}
