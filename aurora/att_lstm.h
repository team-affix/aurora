#pragma once
#include "affix-base/pch.h"
#include "model.h"
#include "sync.h"
#include "att_lstm_ts.h"
#include "lstm.h"

namespace aurora {
	namespace models {
		class att_lstm : public model {
		public:
			size_t m_units = 0;

		public:
			Sync m_models;
			Lstm m_internal_lstm;

		public:
			ATTENTION_FIELDS
			virtual ~att_lstm();
			att_lstm();
			att_lstm(size_t a_units, std::vector<size_t> a_h_dims);

		};
		typedef affix_base::data::ptr<att_lstm> Att_lstm;
	}
}
