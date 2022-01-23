#pragma once
#include "affix-base/pch.h"
#include "stacked_recurrent.h"
#include "sync.h"
#include "lstm.h"
#include "param_vector.h"

namespace aurora {
	namespace pseudo {

		aurora::models::Stacked_recurrent lstm_mdim(
			size_t a_x_units,
			size_t a_h_units,
			size_t a_y_units
		);

		aurora::models::Stacked_recurrent lstm_stacked(
			size_t a_units,
			size_t a_stack_height
		);

		aurora::models::Stacked_recurrent lstm_stacked_mdim(
			size_t a_x_units,
			size_t a_h_units,
			size_t a_y_units,
			size_t a_stack_height
		);

	}
}
