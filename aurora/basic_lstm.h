#pragma once
#include "pch.h"
#include "pseudo.h"
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
	namespace basic {
		Lstm lstm_compiled(size_t a_units, size_t a_max_timesteps, param_vector& a_param_vec);
		Stacked_recurrent lstm_mdim(size_t a_x_units, size_t a_h_units, size_t a_y_units, param_vector& a_param_vec);
		Stacked_recurrent lstm_mdim_compiled(size_t a_x_units, size_t a_h_units, size_t a_y_units, size_t a_max_timesteps, param_vector& a_param_vec);
	}
}
