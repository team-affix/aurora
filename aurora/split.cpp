#include "affix-base/pch.h"
#include "split.h"

using aurora::models::split;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::maths::tensor;

split::~split(

)
{

}

split::split(

)
{

}

split::split(
	const std::vector<size_t>& a_units
) :
	m_units(a_units)
{

}

void split::param_recur(
	const function<void(Param&)>& a_func
)
{

}

model* split::clone(
	const function<Param(Param&)>& a_func
)
{
	split* result = new split();
	result->m_units = m_units;
	return result;
}

void split::fwd(

)
{

}

void split::bwd(

)
{

}

void split::model_recur(
	const function<void(model*)>& a_func
)
{
	a_func(this);
}

void split::compile(

)
{
	int l_x_iterator = 0;

	// Get total units involved in the split
	for (int i = 0; i < m_units.size(); i++)
	{
		x().resize(x().size() + m_units[i]);
		x_grad().resize(x_grad().size() + m_units[i]);
		y().vec().push_back(tensor::new_1d(m_units[i]));
		y_grad().vec().push_back(tensor::new_1d(m_units[i]));

		for (int j = 0; j < m_units[i]; j++)
		{
			// Set equal, to link both val_ptr and vec_ptr
			x()[l_x_iterator] = y()[i][j];

			// Set equal, to link both val_ptr and vec_ptr
			x_grad()[l_x_iterator] = y_grad()[i][j];

			l_x_iterator++;

		}
	}
}
