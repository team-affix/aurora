#include "affix-base/pch.h"
#include "cnl.h"
#include "weight_junction.h"
#include "layer.h"
#include "parameterized_dot_1d.h"

using aurora::models::cnl;
using std::function;
using aurora::params::Param;
using aurora::maths::tensor;
using aurora::models::model;
using aurora::models::layer;

cnl::~cnl()
{

}

cnl::cnl()
{

}

cnl::cnl(
	const size_t& a_filter_height,
	const size_t& a_filter_width,
	const size_t& a_x_stride,
	const size_t& a_y_stride
) :
	m_filter_height(a_filter_height),
	m_filter_width(a_filter_width),
	m_x_stride(a_x_stride),
	m_y_stride(a_y_stride)
{
	m_filters = new sync(new parameterized_dot_1d(m_filter_height * m_filter_width));
}

void cnl::param_recur(
	const function<void(Param&)>& a_func
)
{
	m_filters->param_recur(a_func);
}

model* cnl::clone(
	const function<Param(Param&)>& a_func
)
{
	cnl* result = new cnl();
	result->m_filter_height = m_filter_height;
	result->m_filter_width = m_filter_width;
	result->m_x_stride = m_x_stride;
	result->m_y_stride = m_y_stride;
	result->m_max_input_height = m_max_input_height;
	result->m_max_input_width = m_max_input_width;
	result->m_filters = m_filters->clone(a_func);
	return result;
}

void cnl::fwd()
{
	m_filters->fwd();
}

void cnl::bwd()
{
	m_filters->bwd();
}

void cnl::signal(
	const tensor& a_y_des
)
{
	m_y.sub_2d(a_y_des, m_y_grad);
}

void cnl::model_recur(
	const function<void(model*)>& a_func
)
{
	m_filters->model_recur(a_func);
	a_func(this);
}

void cnl::compile()
{
	m_x = tensor::new_2d(m_max_input_height, m_max_input_width);
	m_x_grad = tensor::new_2d(m_max_input_height, m_max_input_width);
	m_y = tensor::new_2d(y_strides(), x_strides());
	m_y_grad = tensor::new_2d(y_strides(), x_strides());

	m_filters->compile();

	const size_t l_y_strides = y_strides();

 	for (int i = 0; i < m_filters->m_prepared.size(); i++)
	{
		size_t l_row = i / l_y_strides;
		size_t l_col = i % l_y_strides;
		m_x.range_2d(l_row, l_col, m_filter_height, m_filter_width).unroll().group_join_all_ranks(m_filters->m_x[i]);
		m_x_grad.range_2d(l_row, l_col, m_filter_height, m_filter_width).unroll().group_join_all_ranks(m_filters->m_x_grad[i]);
		m_y[l_row][l_col].group_join_all_ranks(m_filters->m_y[i]);
		m_y_grad[l_row][l_col].group_join_all_ranks(m_filters->m_y_grad[i]);
	}

}

void cnl::prep_for_input(
	const size_t& a_input_height,
	const size_t& a_input_width
)
{
	m_max_input_height = a_input_height;
	m_max_input_width = a_input_width;
	m_filters->prep(y_strides(a_input_height) * x_strides(a_input_width));
}

void cnl::unroll_for_input(
	const size_t& a_input_height,
	const size_t& a_input_width
)
{
	assert(a_input_height <= m_max_input_height);
	assert(a_input_width <= m_max_input_width);

	m_filters->unroll(y_strides(a_input_height) * x_strides(a_input_width));

}

size_t cnl::y_strides(
	const size_t& a_height
) const
{
	return (a_height - m_filter_height) / m_y_stride + 1;
}

size_t cnl::x_strides(
	const size_t& a_width
) const
{
	return (a_width - m_filter_width) / m_x_stride + 1;
}

size_t cnl::y_strides() const
{
	return y_strides(m_max_input_height);
}

size_t cnl::x_strides() const 
{
	return x_strides(m_max_input_width);
}
