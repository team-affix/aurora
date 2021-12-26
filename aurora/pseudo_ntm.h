#pragma once
#include "affix-base/pch.h"
#include "ntm.h"
#include "stacked_recurrent.h"
#include "param_vector.h"
#include "ntm.h"

namespace aurora {
	namespace pseudo {
		aurora::models::Ntm ntm_compiled(
			size_t a_memory_height,
			size_t a_memory_width,
			size_t a_num_readers,
			size_t a_num_writers,
			std::vector<int> a_valid_shifts,
			std::vector<size_t> a_head_hidden_dims,
			size_t a_max_timesteps,
			aurora::params::param_vector& a_param_vec
		);

		aurora::models::Stacked_recurrent ntm_mdim(
			size_t a_x_units,
			size_t a_y_units,
			size_t a_memory_height,
			size_t a_memory_width,
			size_t a_num_readers,
			size_t a_num_writers,
			std::vector<int> a_valid_shifts,
			std::vector<size_t> a_head_hidden_dims,
			aurora::params::param_vector& a_param_vec
		);

		aurora::models::Stacked_recurrent ntm_mdim_compiled(
			size_t a_x_units,
			size_t a_y_units,
			size_t a_memory_height,
			size_t a_memory_width,
			size_t a_num_readers,
			size_t a_num_writers,
			std::vector<int> a_valid_shifts,
			std::vector<size_t> a_head_hidden_dims,
			size_t a_max_timesteps,
			aurora::params::param_vector& a_param_vec
		);

	}
}
