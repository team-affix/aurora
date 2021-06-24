#pragma once
#include "pch.h"
#include "ntm.h"
#include "stacked_recurrent.h"
#include "param_vector.h"

using aurora::models::Stacked_recurrent;
using aurora::params::param_vector;

namespace aurora {
	namespace basic {
		Stacked_recurrent ntm_mdim(size_t a_x_units, size_t a_h_units, size_t a_y_units, size_t a_memory_height, size_t a_complexity, size_t a_max_timesteps, param_vector& a_param_vec);
	}
}
