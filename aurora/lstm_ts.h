#pragma once
#include "pch.h"
#include "model.h"
#include "layer.h"
#include "pseudo.h"

using aurora::models::model;

namespace aurora {
	namespace models {
		class lstm_ts : public model {
		public:
			size_t units;

		public:
			tensor ctx;
			tensor cty;
			tensor htx;
			tensor hty;
			tensor ctx_grad;
			tensor cty_grad;
			tensor htx_grad;
			tensor hty_grad;

		public:
			tensor gate_x;
			tensor comp_0;
			tensor comp_1;

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
			lstm_ts(size_t a_units, function<void(ptr<param>&)> a_func);
			lstm_ts(size_t a_units, Layer a_forget_gate, Layer a_limit_gate, Layer a_input_gate, Layer a_output_gate, Layer a_tanh_gate);

		};
		typedef ptr<lstm_ts> Lstm_ts;
	}
}