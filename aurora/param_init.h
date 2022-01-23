#pragma once
#include "affix-base/pch.h"
#include "param.h"

namespace aurora {
	namespace pseudo {

		template<typename PARAM_VECTOR_TYPE>
		std::function<void(aurora::params::Param&)> param_init(const aurora::params::Param& a_param_example, PARAM_VECTOR_TYPE& a_param_vector)
		{
			return [&, a_param_example](aurora::params::Param& a_param)
			{
				a_param = a_param_example->clone();
				a_param_vector.push_back(a_param.get());
			};
		}

		template<typename PARAM_VECTOR_TYPE>
		std::function<void(aurora::params::Param&)> param_dump(PARAM_VECTOR_TYPE& a_param_vector)
		{
			return [&](aurora::params::Param& a_param)
			{
				a_param_vector.push_back(a_param.get());
			};
		}

		template<typename PARAM_COUNTER_TYPE>
		std::function<void(aurora::params::Param&)> param_count(PARAM_COUNTER_TYPE& a_count)
		{
			return [&](aurora::params::Param& a_param)
			{
				a_count++;
			};
		}

	}
}
