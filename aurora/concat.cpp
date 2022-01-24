#include "concat.h"

using aurora::models::concat;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::maths::tensor;

concat::~concat(

)
{

}

concat::concat(

)
{

}

concat::concat(
	const std::vector<size_t>& a_units
) :
	m_units(a_units)
{

}

void concat::param_recur(
	const function<void(Param&)>& a_func
)
{

}

model* concat::clone(
	const function<Param(Param&)>& a_func
)
{
	concat* result = new concat();
	result->m_units = m_units;
	return result;
}

void concat::fwd()
{

}

void concat::bwd()
{

}

void concat::model_recur(
	const function<void(model*)>& a_func
)
{
	a_func(this);
}

void concat::compile()
{
	int l_y_iterator = 0;

	for (int i = 0; i < m_units.size(); i++)
	{
		x().vec().push_back(tensor::new_1d(m_units[i]));
		x_grad().vec().push_back(tensor::new_1d(m_units[i]));
		y().resize(y().size() + m_units[i]);
		y_grad().resize(y_grad().size() + m_units[i]);

		for (int j = 0; j < m_units[i]; j++)
		{
			x()[i][j] = y()[l_y_iterator];
			x_grad()[i][j] = y_grad()[l_y_iterator];
			l_y_iterator++;
		}
	}

}

