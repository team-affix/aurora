#pragma once
#include "pch.h"
#include "ntm.h"
#include "stacked_recurrent.h"
#include "param_vector.h"
#include "ntm.h"

using aurora::models::Stacked_recurrent;
using aurora::params::param_vector;
using aurora::models::Ntm;

namespace aurora {
	namespace pseudo {
		Ntm ntm_compiled(
			size_t a_memory_height,
			size_t a_memory_width,
			size_t a_num_readers,
			size_t a_num_writers,
			vector<int> a_valid_shifts,
			vector<size_t> a_head_hidden_dims,
			size_t a_max_timesteps,
			param_vector& a_param_vec);

		Stacked_recurrent ntm_mdim(
			size_t a_x_units,
			size_t a_y_units,
			size_t a_memory_height,
			size_t a_memory_width,
			size_t a_num_readers,
			size_t a_num_writers,
			vector<int> a_valid_shifts,
			vector<size_t> a_head_hidden_dims,
			param_vector& a_param_vec);

		Stacked_recurrent ntm_mdim_compiled(
			size_t a_x_units,
			size_t a_y_units,
			size_t a_memory_height,
			size_t a_memory_width,
			size_t a_num_readers,
			size_t a_num_writers,
			vector<int> a_valid_shifts,
			vector<size_t> a_head_hidden_dims,
			size_t a_max_timesteps,
			param_vector& a_param_vec);

	}
}
