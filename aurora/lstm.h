#pragma once
#include "pch.h"
#include "data.h"
#include "model.h"
#include "lstm_ts.h"

using affix_base::data::ptr;
using aurora::models::model;
using aurora::models::lstm_ts;

namespace aurora {
	namespace models {
		class lstm : public model {
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