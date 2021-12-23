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
			tensor ctx;
			tensor cty;
			tensor htx;
			tensor hty;
			tensor ctx_grad;
			tensor cty_grad;
			tensor htx_grad;
			tensor hty_grad;

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
		typedef ptr<lstm> Lstm;
	}
}