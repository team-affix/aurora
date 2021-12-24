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
			size_t units = 0;

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
			Lstm_ts lstm_ts_template;
			vector<Lstm_ts> prepared;
			vector<Lstm_ts> unrolled;

		public:
			RECURRENT_FIELDS
			virtual ~lstm();
			lstm();
			lstm(size_t a_units);

		};
		typedef affix_base::data::ptr<lstm> Lstm;
	}
}
