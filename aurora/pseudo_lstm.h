#pragma once
#include "affix-base/pch.h"
#include "stacked_recurrent.h"
#include "sync.h"
#include "lstm.h"
#include "param_vector.h"

namespace aurora {
	namespace pseudo {
		aurora::models::Lstm lstm_compiled(
			size_t a_units,
			size_t a_max_timesteps,
			aurora::params::param_vector& a_param_vec
		);
		aurora::models::Stacked_recurrent lstm_mdim(
			size_t a_x_units,
			size_t a_h_units,
			size_t a_y_units
		);
		aurora::models::Stacked_recurrent lstm_mdim_compiled(
			size_t a_x_units,
			size_t a_h_units,
			size_t a_y_units,
			size_t a_max_timesteps,
			aurora::params::param_vector& a_param_vec
		);
		aurora::models::Stacked_recurrent lstm_stacked(
			size_t a_units,
			size_t a_stack_height
		);
		aurora::models::Stacked_recurrent lstm_stacked_compiled(
			size_t a_units,
			size_t a_stack_height,
			size_t a_max_timesteps,
			aurora::params::param_vector& a_param_vec
		);
		aurora::models::Stacked_recurrent lstm_stacked_mdim(
			size_t a_x_units,
			size_t a_h_units,
			size_t a_y_units,
			size_t a_stack_height
		);
		aurora::models::Stacked_recurrent lstm_stacked_mdim_compiled(
			size_t a_x_units,
			size_t a_h_units,
			size_t a_y_units,
			size_t a_stack_height,
			size_t a_max_timesteps,
			aurora::params::param_vector& a_param_vec
		);
	}
}
