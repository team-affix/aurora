#pragma once
#include "affix-base/pch.h"
#include "model.h"
#include "sync.h"
#include "pseudo_tnn.h"
#include "neuron.h"

namespace aurora {
	namespace models {
		class att_lstm_ts : public recurrent {
		public:
			size_t units;

		public:
			aurora::maths::tensor htx;
			aurora::maths::tensor htx_grad;

		public:
			Model model_template;
			Sync models;

		public:
			RECURRENT_FIELDS
			virtual ~att_lstm_ts();
			att_lstm_ts();
			att_lstm_ts(size_t a_units, vector<size_t> a_h_dims);

		};
		typedef ptr<att_lstm_ts> Att_lstm_ts;
	}
}