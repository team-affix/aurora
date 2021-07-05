#pragma once
#include "pch.h"
#include "stacked_recurrent.h"
#include "sync.h"
#include "lstm.h"
#include "param_vector.h"

using aurora::models::stacked_recurrent;
using aurora::models::Stacked_recurrent;
using aurora::models::sync;
using aurora::models::Lstm;
using aurora::models::lstm;

namespace aurora {
	namespace pseudo {
		Lstm lstm_compiled(size_t a_units, size_t a_max_timesteps, param_vector& a_param_vec);
		Stacked_recurrent lstm_mdim(size_t a_x_units, size_t a_h_units, size_t a_y_units);
		Stacked_recurrent lstm_mdim_compiled(size_t a_x_units, size_t a_h_units, size_t a_y_units, size_t a_max_timesteps, param_vector& a_param_vec);
		Stacked_recurrent lstm_stacked(size_t a_units, size_t a_stack_height);
		Stacked_recurrent lstm_stacked_compiled(size_t a_units, size_t a_stack_height, size_t a_max_timesteps, param_vector& a_param_vec);
		Stacked_recurrent lstm_stacked_mdim(size_t a_x_units, size_t a_h_units, size_t a_y_units, size_t a_stack_height);
		Stacked_recurrent lstm_stacked_mdim_compiled(size_t a_x_units, size_t a_h_units, size_t a_y_units, size_t a_stack_height, size_t a_max_timesteps, param_vector& a_param_vec);
	}
}
