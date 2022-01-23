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
			size_t m_units;

		public:
			aurora::maths::tensor m_htx;
			aurora::maths::tensor m_htx_grad;

		public:
			Model m_model_template;
			Sync m_models;

		public:
			RECURRENT_FIELDS
			virtual ~att_lstm_ts();
			att_lstm_ts();
			att_lstm_ts(
				size_t a_units,
				std::vector<size_t> a_h_dims
			);

		};
		typedef affix_base::data::ptr<att_lstm_ts> Att_lstm_ts;
	}
}
