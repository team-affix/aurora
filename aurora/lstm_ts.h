#pragma once
#include "affix-base/pch.h"
#include "model.h"
#include "layer.h"
#include "pseudo_tnn.h"

namespace aurora {
	namespace models {
		class lstm_ts : public model {
		public:
			size_t units;

		public:
			aurora::maths::tensor ctx;
			aurora::maths::tensor cty;
			aurora::maths::tensor htx;
			aurora::maths::tensor hty;
			aurora::maths::tensor ctx_grad;
			aurora::maths::tensor cty_grad;
			aurora::maths::tensor htx_grad;
			aurora::maths::tensor hty_grad;

		public:
			aurora::maths::tensor gate_x;
			aurora::maths::tensor comp_0;
			aurora::maths::tensor comp_1;

		public:
			Layer forget_gate;
			Layer limit_gate;
			Layer input_gate;
			Layer output_gate;
			Layer tanh_gate;

		public:
			MODEL_FIELDS
			virtual ~lstm_ts();
			lstm_ts();
			lstm_ts(size_t a_units);
			lstm_ts(size_t a_units, Layer a_forget_gate, Layer a_limit_gate, Layer a_input_gate, Layer a_output_gate, Layer a_tanh_gate);

		};
		typedef affix_base::data::ptr<lstm_ts> Lstm_ts;
	}
}
