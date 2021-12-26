#pragma once
#include "affix-base/pch.h"
#include "affix-base/data.h"
#include "model.h"
#include "recurrent.h"
#include "lstm_ts.h"

namespace aurora {
	namespace models {
		class lstm : public recurrent {
		public:
			size_t m_units = 0;

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
			Lstm_ts m_lstm_ts_template;
			std::vector<Lstm_ts> m_prepared;
			std::vector<Lstm_ts> m_unrolled;

		public:
			RECURRENT_FIELDS
			virtual ~lstm();
			lstm();
			lstm(size_t a_units);

		};
		typedef affix_base::data::ptr<lstm> Lstm;
	}
}
