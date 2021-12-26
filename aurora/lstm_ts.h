#pragma once
#include "affix-base/pch.h"
#include "model.h"
#include "layer.h"
#include "pseudo_tnn.h"

namespace aurora {
	namespace models {
		class lstm_ts : public model {
		public:
			size_t m_units;

		public:
			aurora::maths::tensor m_ctx;
			aurora::maths::tensor m_cty;
			aurora::maths::tensor m_htx;
			aurora::maths::tensor m_hty;
			aurora::maths::tensor m_ctx_grad;
			aurora::maths::tensor m_cty_grad;
			aurora::maths::tensor m_htx_grad;
			aurora::maths::tensor m_hty_grad;

		public:
			aurora::maths::tensor m_gate_x;
			aurora::maths::tensor m_comp_0;
			aurora::maths::tensor m_comp_1;

		public:
			Layer m_forget_gate;
			Layer m_limit_gate;
			Layer m_input_gate;
			Layer m_output_gate;
			Layer m_tanh_gate;

		public:
			MODEL_FIELDS
			virtual ~lstm_ts();
			lstm_ts();
			lstm_ts(
				size_t a_units
			);
			lstm_ts(
				size_t a_units,
				Layer a_forget_gate,
				Layer a_limit_gate,
				Layer a_input_gate,
				Layer a_output_gate,
				Layer a_tanh_gate
			);

		};
		typedef affix_base::data::ptr<lstm_ts> Lstm_ts;
	}
}
